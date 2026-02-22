#include "MPIBarnesHut.h"
#include "MortonCurve.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

// Gravitational constant
constexpr double G = 6.67430e-11;
constexpr double MIN_DISTANCE = 0.5;

MPIBarnesHut::MPIBarnesHut(double theta, InteractionForce& interactionForce, MPI_Comm comm)
    : m_comm(comm)
    , m_rank(0)
    , m_numRanks(1)
    , m_theta(theta)
    , m_interactionForce(interactionForce)
    , m_localParticleCount(0)
    , m_globalMinBound{0, 0, 0}
    , m_globalMaxBound{0, 0, 0}
{
    MPI_Comm_rank(m_comm, &m_rank);
    MPI_Comm_size(m_comm, &m_numRanks);
}

MPIBarnesHut::~MPIBarnesHut() {
    // Nothing to clean up - MPI resources managed externally
}

void MPIBarnesHut::computeGlobalBounds(const ParticleData& particles) {
    // Find local bounds
    std::array<double, 3> localMin = {
        std::numeric_limits<double>::max(),
        std::numeric_limits<double>::max(),
        std::numeric_limits<double>::max()
    };
    std::array<double, 3> localMax = {
        std::numeric_limits<double>::lowest(),
        std::numeric_limits<double>::lowest(),
        std::numeric_limits<double>::lowest()
    };

    size_t n = particles.size();
    for (size_t i = 0; i < n; ++i) {
        localMin[0] = std::min(localMin[0], particles.posX[i]);
        localMin[1] = std::min(localMin[1], particles.posY[i]);
        localMin[2] = std::min(localMin[2], particles.posZ[i]);
        localMax[0] = std::max(localMax[0], particles.posX[i]);
        localMax[1] = std::max(localMax[1], particles.posY[i]);
        localMax[2] = std::max(localMax[2], particles.posZ[i]);
    }

    // All-reduce to get global bounds
    MPI_Allreduce(localMin.data(), m_globalMinBound.data(), 3, MPI_DOUBLE, MPI_MIN, m_comm);
    MPI_Allreduce(localMax.data(), m_globalMaxBound.data(), 3, MPI_DOUBLE, MPI_MAX, m_comm);

    // Add small padding to avoid boundary issues
    double padding = 0.01 * std::max({
        m_globalMaxBound[0] - m_globalMinBound[0],
        m_globalMaxBound[1] - m_globalMinBound[1],
        m_globalMaxBound[2] - m_globalMinBound[2]
    });
    
    for (int d = 0; d < 3; ++d) {
        m_globalMinBound[d] -= padding;
        m_globalMaxBound[d] += padding;
    }
}

void MPIBarnesHut::partitionParticles(const ParticleData& particles) {
    size_t n = particles.size();
    
    // Compute Morton codes for all particles
    auto mortonSorted = MortonCurve::sortParticlesByMorton(
        particles.posX, particles.posY, particles.posZ,
        m_globalMinBound, m_globalMaxBound);

    // Compute partitions for each rank
    auto partitions = MortonCurve::computeDomainPartitions(mortonSorted, m_numRanks);

    // Extract local indices for this rank
    m_localIndices.clear();
    if (m_rank < static_cast<int>(partitions.size())) {
        size_t startIdx = partitions[m_rank].first;
        size_t count = partitions[m_rank].second;
        m_localIndices.reserve(count);
        
        for (size_t i = 0; i < count; ++i) {
            m_localIndices.push_back(mortonSorted[startIdx + i].second);
        }
    }
    
    m_localParticleCount = m_localIndices.size();
}

void MPIBarnesHut::buildLocalTree(const ParticleData& particles) {
    // For the simplified implementation, we don't build a full tree
    // Instead, we aggregate particles into pseudo-particles at specified levels
    // This is sufficient for the LET approach
}

std::vector<MPIBarnesHut::PseudoParticle> MPIBarnesHut::aggregateLocalParticles(
    const ParticleData& particles, int level) {
    
    // Number of cells at this level is 8^level
    size_t numCells = 1 << (3 * level);  // 2^(3*level) = 8^level
    
    // Use Morton code prefix to bucket particles
    std::vector<std::array<double, 3>> weightedPos(numCells, {0, 0, 0});
    std::vector<double> totalMass(numCells, 0);
    std::vector<int> particleCount(numCells, 0);
    
    uint64_t levelMask = MortonCurve::getLevelMask(level);
    
    for (size_t localIdx = 0; localIdx < m_localIndices.size(); ++localIdx) {
        size_t i = m_localIndices[localIdx];
        std::array<double, 3> pos = {particles.posX[i], particles.posY[i], particles.posZ[i]};
        
        uint64_t mortonCode = MortonCurve::encodePosition(pos, m_globalMinBound, m_globalMaxBound);
        
        // Extract bucket index from Morton code at this level
        size_t bucketIdx = 0;
        for (int l = 0; l < level; ++l) {
            int octant = MortonCurve::getOctantAtLevel(mortonCode, l);
            bucketIdx = bucketIdx * 8 + octant;
        }
        
        if (bucketIdx < numCells) {
            double mass = particles.mass[i];
            weightedPos[bucketIdx][0] += pos[0] * mass;
            weightedPos[bucketIdx][1] += pos[1] * mass;
            weightedPos[bucketIdx][2] += pos[2] * mass;
            totalMass[bucketIdx] += mass;
            particleCount[bucketIdx]++;
        }
    }
    
    // Create pseudo-particles from non-empty buckets
    std::vector<PseudoParticle> pseudos;
    
    double domainSize = std::max({
        m_globalMaxBound[0] - m_globalMinBound[0],
        m_globalMaxBound[1] - m_globalMinBound[1],
        m_globalMaxBound[2] - m_globalMinBound[2]
    });
    double cellSize = domainSize / (1 << level);
    
    for (size_t idx = 0; idx < numCells; ++idx) {
        if (totalMass[idx] > 0) {
            PseudoParticle p;
            p.centerOfMass[0] = weightedPos[idx][0] / totalMass[idx];
            p.centerOfMass[1] = weightedPos[idx][1] / totalMass[idx];
            p.centerOfMass[2] = weightedPos[idx][2] / totalMass[idx];
            p.totalMass = totalMass[idx];
            p.size = cellSize / 2.0;
            pseudos.push_back(p);
        }
    }
    
    return pseudos;
}

void MPIBarnesHut::exchangeEssentialTree(const ParticleData& particles) {
    // Aggregate local particles into pseudo-particles at level 2 (64 cells)
    constexpr int AGGREGATION_LEVEL = 2;
    auto localPseudos = aggregateLocalParticles(particles, AGGREGATION_LEVEL);
    
    // Exchange counts first
    int localCount = static_cast<int>(localPseudos.size());
    std::vector<int> allCounts(m_numRanks);
    MPI_Allgather(&localCount, 1, MPI_INT, allCounts.data(), 1, MPI_INT, m_comm);
    
    // Calculate displacements for gathering
    std::vector<int> displacements(m_numRanks);
    int totalPseudos = 0;
    for (int r = 0; r < m_numRanks; ++r) {
        displacements[r] = totalPseudos;
        totalPseudos += allCounts[r];
    }
    
    // Pack pseudo-particles into flat arrays for MPI
    // Each pseudo-particle: 3 doubles (COM) + 1 double (mass) + 1 double (size) = 5 doubles
    constexpr int DOUBLES_PER_PSEUDO = 5;
    
    std::vector<double> localData(localCount * DOUBLES_PER_PSEUDO);
    for (int i = 0; i < localCount; ++i) {
        localData[i * DOUBLES_PER_PSEUDO + 0] = localPseudos[i].centerOfMass[0];
        localData[i * DOUBLES_PER_PSEUDO + 1] = localPseudos[i].centerOfMass[1];
        localData[i * DOUBLES_PER_PSEUDO + 2] = localPseudos[i].centerOfMass[2];
        localData[i * DOUBLES_PER_PSEUDO + 3] = localPseudos[i].totalMass;
        localData[i * DOUBLES_PER_PSEUDO + 4] = localPseudos[i].size;
    }
    
    // Allgatherv to exchange pseudo-particles
    std::vector<int> byteCounts(m_numRanks), byteDisplacements(m_numRanks);
    for (int r = 0; r < m_numRanks; ++r) {
        byteCounts[r] = allCounts[r] * DOUBLES_PER_PSEUDO;
        byteDisplacements[r] = displacements[r] * DOUBLES_PER_PSEUDO;
    }
    
    std::vector<double> allData(totalPseudos * DOUBLES_PER_PSEUDO);
    MPI_Allgatherv(localData.data(), localCount * DOUBLES_PER_PSEUDO, MPI_DOUBLE,
                   allData.data(), byteCounts.data(), byteDisplacements.data(), MPI_DOUBLE, m_comm);
    
    // Unpack into remote pseudo-particles (exclude our own)
    m_remotePseudoParticles.clear();
    m_remotePseudoParticles.reserve(totalPseudos - localCount);
    
    for (int r = 0; r < m_numRanks; ++r) {
        if (r == m_rank) continue;  // Skip our own
        
        int offset = displacements[r];
        for (int i = 0; i < allCounts[r]; ++i) {
            PseudoParticle p;
            int baseIdx = (offset + i) * DOUBLES_PER_PSEUDO;
            p.centerOfMass[0] = allData[baseIdx + 0];
            p.centerOfMass[1] = allData[baseIdx + 1];
            p.centerOfMass[2] = allData[baseIdx + 2];
            p.totalMass = allData[baseIdx + 3];
            p.size = allData[baseIdx + 4];
            m_remotePseudoParticles.push_back(p);
        }
    }
}

bool MPIBarnesHut::isSufficientlyFar(const PseudoParticle& pseudo,
                                      const std::array<double, 3>& particlePos) const {
    double dx = pseudo.centerOfMass[0] - particlePos[0];
    double dy = pseudo.centerOfMass[1] - particlePos[1];
    double dz = pseudo.centerOfMass[2] - particlePos[2];
    double distSq = dx * dx + dy * dy + dz * dz;
    
    if (distSq < 1e-10) return false;
    
    double regionWidth = pseudo.size * 2.0;
    return (regionWidth * regionWidth / distSq) < m_theta;
}

std::array<double, 3> MPIBarnesHut::calculatePseudoForce(
    const std::array<double, 3>& position, double mass,
    const PseudoParticle& pseudo) {
    
    std::array<double, 3> force = {0, 0, 0};
    
    double dx = pseudo.centerOfMass[0] - position[0];
    double dy = pseudo.centerOfMass[1] - position[1];
    double dz = pseudo.centerOfMass[2] - position[2];
    double distSq = dx * dx + dy * dy + dz * dz;
    double dist = std::sqrt(distSq);
    
    if (dist < MIN_DISTANCE) return force;
    
    double forceMag = (G * mass * pseudo.totalMass) / distSq;
    double invDist = 1.0 / dist;
    
    force[0] = forceMag * dx * invDist;
    force[1] = forceMag * dy * invDist;
    force[2] = forceMag * dz * invDist;
    
    return force;
}

void MPIBarnesHut::calculateLocalForces(const ParticleData& particles,
                                         std::vector<double>& forceX,
                                         std::vector<double>& forceY,
                                         std::vector<double>& forceZ) {
    size_t n = particles.size();
    forceX.assign(n, 0.0);
    forceY.assign(n, 0.0);
    forceZ.assign(n, 0.0);
    
    // For each local particle
    #pragma omp parallel for schedule(dynamic)
    for (size_t localIdx = 0; localIdx < m_localIndices.size(); ++localIdx) {
        size_t i = m_localIndices[localIdx];
        
        std::array<double, 3> pos = {particles.posX[i], particles.posY[i], particles.posZ[i]};
        double mass = particles.mass[i];
        
        double fx = 0, fy = 0, fz = 0;
        
        // 1. Calculate forces from other local particles (direct N-body for local domain)
        for (size_t otherLocalIdx = 0; otherLocalIdx < m_localIndices.size(); ++otherLocalIdx) {
            if (otherLocalIdx == localIdx) continue;
            
            size_t j = m_localIndices[otherLocalIdx];
            std::array<double, 3> otherPos = {particles.posX[j], particles.posY[j], particles.posZ[j]};
            
            auto force = m_interactionForce.calculateForceSOA(pos, mass, otherPos, particles.mass[j]);
            fx += force[0];
            fy += force[1];
            fz += force[2];
        }
        
        // 2. Calculate forces from remote pseudo-particles (approximation)
        for (const auto& pseudo : m_remotePseudoParticles) {
            // Use pseudo-particle approximation (always, since it's already aggregated)
            auto force = calculatePseudoForce(pos, mass, pseudo);
            fx += force[0];
            fy += force[1];
            fz += force[2];
        }
        
        forceX[i] = fx;
        forceY[i] = fy;
        forceZ[i] = fz;
    }
}

void MPIBarnesHut::run(const ParticleData& particles,
                       std::vector<double>& forceX,
                       std::vector<double>& forceY,
                       std::vector<double>& forceZ) {
    // Step 1: Compute global bounding box
    computeGlobalBounds(particles);
    
    // Step 2: Partition particles using Morton curve
    partitionParticles(particles);
    
    // Step 3: Exchange essential tree data
    exchangeEssentialTree(particles);
    
    // Step 4: Calculate forces for local particles
    calculateLocalForces(particles, forceX, forceY, forceZ);
    
    // Step 5: Allreduce forces so all ranks have complete force data
    // (In a real application, you'd only update local particles)
    std::vector<double> globalForceX(particles.size(), 0.0);
    std::vector<double> globalForceY(particles.size(), 0.0);
    std::vector<double> globalForceZ(particles.size(), 0.0);
    
    MPI_Allreduce(forceX.data(), globalForceX.data(), particles.size(), MPI_DOUBLE, MPI_SUM, m_comm);
    MPI_Allreduce(forceY.data(), globalForceY.data(), particles.size(), MPI_DOUBLE, MPI_SUM, m_comm);
    MPI_Allreduce(forceZ.data(), globalForceZ.data(), particles.size(), MPI_DOUBLE, MPI_SUM, m_comm);
    
    forceX = std::move(globalForceX);
    forceY = std::move(globalForceY);
    forceZ = std::move(globalForceZ);
}

void gatherAllForces(const std::vector<double>& localForceX,
                     const std::vector<double>& localForceY,
                     const std::vector<double>& localForceZ,
                     std::vector<double>& globalForceX,
                     std::vector<double>& globalForceY,
                     std::vector<double>& globalForceZ,
                     const std::vector<int>& counts,
                     const std::vector<int>& displacements,
                     MPI_Comm comm) {
    int totalSize = 0;
    for (int c : counts) totalSize += c;
    
    globalForceX.resize(totalSize);
    globalForceY.resize(totalSize);
    globalForceZ.resize(totalSize);
    
    MPI_Allgatherv(localForceX.data(), localForceX.size(), MPI_DOUBLE,
                   globalForceX.data(), counts.data(), displacements.data(), MPI_DOUBLE, comm);
    MPI_Allgatherv(localForceY.data(), localForceY.size(), MPI_DOUBLE,
                   globalForceY.data(), counts.data(), displacements.data(), MPI_DOUBLE, comm);
    MPI_Allgatherv(localForceZ.data(), localForceZ.size(), MPI_DOUBLE,
                   globalForceZ.data(), counts.data(), displacements.data(), MPI_DOUBLE, comm);
}

void updateParticlesMPI(ParticleData& particles,
                        const std::vector<double>& forceX,
                        const std::vector<double>& forceY,
                        const std::vector<double>& forceZ,
                        double timeStep,
                        const std::vector<size_t>& localIndices) {
    #pragma omp parallel for
    for (size_t localIdx = 0; localIdx < localIndices.size(); ++localIdx) {
        size_t i = localIndices[localIdx];
        double mass = particles.mass[i];
        if (mass <= 0) continue;
        
        double invMass = 1.0 / mass;
        double ax = forceX[i] * invMass;
        double ay = forceY[i] * invMass;
        double az = forceZ[i] * invMass;
        
        particles.velX[i] += ax * timeStep;
        particles.velY[i] += ay * timeStep;
        particles.velZ[i] += az * timeStep;
        
        particles.posX[i] += particles.velX[i] * timeStep;
        particles.posY[i] += particles.velY[i] * timeStep;
        particles.posZ[i] += particles.velZ[i] * timeStep;
    }
}
