#include "MortonCurve.h"
#include <cmath>
#include <stdexcept>

/**
 * @brief Spread bits of a 21-bit integer into a 63-bit value with 2 zeros between each bit
 * 
 * Input:  00000000000xxxxxxxxxxxxxxxxxxxxxxx (21 bits)
 * Output: 00x00x00x00x00x00x00x00x00x00x00x00x00x00x00x00x00x00x00x00x00x (63 bits)
 */
uint64_t MortonCurve::spreadBits3D(uint32_t v) {
    uint64_t x = v & MAX_COORD;  // Mask to 21 bits
    
    // Spread bits using magic numbers
    // This uses a divide-and-conquer approach
    x = (x | (x << 32)) & 0x001F00000000FFFFULL;
    x = (x | (x << 16)) & 0x001F0000FF0000FFULL;
    x = (x | (x << 8))  & 0x100F00F00F00F00FULL;
    x = (x | (x << 4))  & 0x10C30C30C30C30C3ULL;
    x = (x | (x << 2))  & 0x1249249249249249ULL;
    
    return x;
}

/**
 * @brief Compact bits from a 63-bit value back to a 21-bit integer
 * 
 * Reverse of spreadBits3D
 */
uint32_t MortonCurve::compactBits3D(uint64_t v) {
    v &= 0x1249249249249249ULL;
    v = (v | (v >> 2))  & 0x10C30C30C30C30C3ULL;
    v = (v | (v >> 4))  & 0x100F00F00F00F00FULL;
    v = (v | (v >> 8))  & 0x001F0000FF0000FFULL;
    v = (v | (v >> 16)) & 0x001F00000000FFFFULL;
    v = (v | (v >> 32)) & 0x00000000001FFFFFULL;
    
    return static_cast<uint32_t>(v);
}

uint64_t MortonCurve::encode(uint32_t x, uint32_t y, uint32_t z) {
    return spreadBits3D(x) | (spreadBits3D(y) << 1) | (spreadBits3D(z) << 2);
}

std::array<uint32_t, 3> MortonCurve::decode(uint64_t code) {
    return {
        compactBits3D(code),
        compactBits3D(code >> 1),
        compactBits3D(code >> 2)
    };
}

uint64_t MortonCurve::encodeNormalized(double x, double y, double z, unsigned int resolution) {
    if (resolution > MAX_RESOLUTION) {
        resolution = MAX_RESOLUTION;
    }
    
    // Scale to integer coordinates
    uint64_t scale = (1ULL << resolution) - 1;
    
    // Clamp coordinates to [0, 1]
    x = std::max(0.0, std::min(1.0, x));
    y = std::max(0.0, std::min(1.0, y));
    z = std::max(0.0, std::min(1.0, z));
    
    uint32_t ix = static_cast<uint32_t>(x * scale);
    uint32_t iy = static_cast<uint32_t>(y * scale);
    uint32_t iz = static_cast<uint32_t>(z * scale);
    
    return encode(ix, iy, iz);
}

uint64_t MortonCurve::encodePosition(
    const std::array<double, 3>& position,
    const std::array<double, 3>& minBound,
    const std::array<double, 3>& maxBound,
    unsigned int resolution) {
    
    // Normalize position to [0, 1] within bounding box
    double rangeX = maxBound[0] - minBound[0];
    double rangeY = maxBound[1] - minBound[1];
    double rangeZ = maxBound[2] - minBound[2];
    
    // Avoid division by zero
    if (rangeX <= 0) rangeX = 1.0;
    if (rangeY <= 0) rangeY = 1.0;
    if (rangeZ <= 0) rangeZ = 1.0;
    
    double normX = (position[0] - minBound[0]) / rangeX;
    double normY = (position[1] - minBound[1]) / rangeY;
    double normZ = (position[2] - minBound[2]) / rangeZ;
    
    return encodeNormalized(normX, normY, normZ, resolution);
}

std::vector<std::pair<uint64_t, size_t>> MortonCurve::sortParticlesByMorton(
    const std::vector<double>& posX,
    const std::vector<double>& posY,
    const std::vector<double>& posZ,
    const std::array<double, 3>& minBound,
    const std::array<double, 3>& maxBound) {
    
    size_t n = posX.size();
    std::vector<std::pair<uint64_t, size_t>> mortonPairs;
    mortonPairs.reserve(n);
    
    // Compute Morton codes for all particles
    for (size_t i = 0; i < n; ++i) {
        std::array<double, 3> pos = {posX[i], posY[i], posZ[i]};
        uint64_t code = encodePosition(pos, minBound, maxBound);
        mortonPairs.emplace_back(code, i);
    }
    
    // Sort by Morton code
    std::sort(mortonPairs.begin(), mortonPairs.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    
    return mortonPairs;
}

std::vector<std::pair<size_t, size_t>> MortonCurve::computeDomainPartitions(
    const std::vector<std::pair<uint64_t, size_t>>& mortonSorted,
    int numRanks) {
    
    size_t n = mortonSorted.size();
    std::vector<std::pair<size_t, size_t>> partitions(numRanks);
    
    if (n == 0 || numRanks <= 0) {
        return partitions;
    }
    
    // Simple equal partitioning based on particle count
    size_t particlesPerRank = n / numRanks;
    size_t remainder = n % numRanks;
    
    size_t offset = 0;
    for (int rank = 0; rank < numRanks; ++rank) {
        size_t count = particlesPerRank + (rank < static_cast<int>(remainder) ? 1 : 0);
        partitions[rank] = {offset, count};
        offset += count;
    }
    
    return partitions;
}

uint64_t MortonCurve::getLevelMask(int level) {
    // Each level uses 3 bits (one for each dimension)
    // Level 0: bits 60-62 (root)
    // Level 1: bits 57-59
    // etc.
    if (level < 0 || level >= static_cast<int>(MAX_RESOLUTION)) {
        return 0;
    }
    
    // Mask for highest (63 - level*3) bits
    int shift = 63 - (level + 1) * 3;
    if (shift < 0) shift = 0;
    
    return ~((1ULL << shift) - 1);
}

int MortonCurve::getOctantAtLevel(uint64_t code, int level) {
    // Extract 3 bits at the specified level
    int shift = 63 - (level + 1) * 3;
    if (shift < 0) shift = 0;
    
    return static_cast<int>((code >> shift) & 0x7);
}
