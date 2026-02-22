#ifndef MORTON_CURVE_H
#define MORTON_CURVE_H

#include <cstdint>
#include <array>
#include <vector>
#include <algorithm>
#include <limits>

/**
 * @brief Morton (Z-order) space-filling curve implementation for 3D domain decomposition
 * 
 * The Morton curve provides a mapping from 3D coordinates to a 1D index while
 * preserving spatial locality. This is crucial for efficient domain decomposition
 * in the MPI-parallelized Barnes-Hut algorithm (Jülich approach).
 * 
 * Morton codes are computed by interleaving the bits of x, y, z coordinates:
 * For coordinates (x, y, z), the Morton code M is:
 * M = ...z2y2x2z1y1x1z0y0x0
 */
class MortonCurve {
public:
    /**
     * @brief Compute Morton code for 3D integer coordinates
     * @param x X coordinate (0 to 2^21-1)
     * @param y Y coordinate (0 to 2^21-1)
     * @param z Z coordinate (0 to 2^21-1)
     * @return 64-bit Morton code
     */
    static uint64_t encode(uint32_t x, uint32_t y, uint32_t z);

    /**
     * @brief Decode Morton code back to 3D integer coordinates
     * @param code 64-bit Morton code
     * @return Array of {x, y, z} coordinates
     */
    static std::array<uint32_t, 3> decode(uint64_t code);

    /**
     * @brief Compute Morton code for normalized floating-point coordinates
     * @param x X coordinate (0.0 to 1.0)
     * @param y Y coordinate (0.0 to 1.0)
     * @param z Z coordinate (0.0 to 1.0)
     * @param resolution Number of bits per dimension (max 21 for 64-bit Morton code)
     * @return 64-bit Morton code
     */
    static uint64_t encodeNormalized(double x, double y, double z, unsigned int resolution = 21);

    /**
     * @brief Compute Morton code for a position within a bounding box
     * @param position 3D position
     * @param minBound Minimum corner of bounding box
     * @param maxBound Maximum corner of bounding box
     * @param resolution Number of bits per dimension
     * @return 64-bit Morton code
     */
    static uint64_t encodePosition(
        const std::array<double, 3>& position,
        const std::array<double, 3>& minBound,
        const std::array<double, 3>& maxBound,
        unsigned int resolution = 21);

    /**
     * @brief Sort particle indices by Morton code
     * @param posX X positions
     * @param posY Y positions
     * @param posZ Z positions
     * @param minBound Minimum corner of bounding box
     * @param maxBound Maximum corner of bounding box
     * @return Vector of sorted (morton_code, original_index) pairs
     */
    static std::vector<std::pair<uint64_t, size_t>> sortParticlesByMorton(
        const std::vector<double>& posX,
        const std::vector<double>& posY,
        const std::vector<double>& posZ,
        const std::array<double, 3>& minBound,
        const std::array<double, 3>& maxBound);

    /**
     * @brief Compute domain boundaries for MPI ranks using Morton-ordered particles
     * @param mortonSorted Sorted vector of (morton_code, index) pairs
     * @param numRanks Number of MPI ranks
     * @return Vector of (start_index, count) pairs for each rank
     */
    static std::vector<std::pair<size_t, size_t>> computeDomainPartitions(
        const std::vector<std::pair<uint64_t, size_t>>& mortonSorted,
        int numRanks);

    /**
     * @brief Get Morton code prefix mask for a given tree level
     * @param level Octree level (0 = root)
     * @return Bitmask for extracting Morton prefix at that level
     */
    static uint64_t getLevelMask(int level);

    /**
     * @brief Get the octant index (0-7) from Morton code at a given level
     * @param code Morton code
     * @param level Octree level
     * @return Octant index (0-7)
     */
    static int getOctantAtLevel(uint64_t code, int level);

private:
    // Bit spreading functions for Morton encoding
    static uint64_t spreadBits3D(uint32_t v);
    static uint32_t compactBits3D(uint64_t v);

    // Maximum resolution (21 bits per dimension = 63 bits total for 3D)
    static constexpr unsigned int MAX_RESOLUTION = 21;
    static constexpr uint64_t MAX_COORD = (1ULL << MAX_RESOLUTION) - 1;
};

/**
 * @brief Structure holding Morton-sorted particle data with domain info
 */
struct MortonParticleData {
    std::vector<uint64_t> mortonCodes;      // Morton codes for each particle
    std::vector<size_t> sortedIndices;       // Original indices in sorted order
    std::array<double, 3> minBound;          // Domain minimum bound
    std::array<double, 3> maxBound;          // Domain maximum bound
    
    size_t size() const { return mortonCodes.size(); }
};

#endif // MORTON_CURVE_H
