#ifndef MPI_BARNES_HUT_H
#define MPI_BARNES_HUT_H

#include <mpi.h>
#include <vector>
#include <array>
#include <memory>
#include "ParticleData.h"
#include "MortonCurve.h"
#include "Cube.h"
#include "InteractionForce.h"

/**
 * @brief MPI-parallelized Barnes-Hut algorithm using space-filling curves (Jülich approach)
 * 
 * This implementation follows the domain decomposition strategy used at Jülich Supercomputing Centre:
 * 1. Morton curve ordering for spatial locality
 * 2. Equal-work domain decomposition along the Morton curve
 * 3. Local Essential Tree (LET) concept for efficient communication
 * 4. Parallel force calculation with minimal communication overhead
 * 
 * Key concepts:
 * - Each MPI rank owns a contiguous segment of the Morton-ordered particles
 * - Ranks exchange "essential" tree information needed for force calculations
 * - Local tree is augmented with pseudo-particles from other ranks
 */
class MPIBarnesHut {
public:
    /**
     * @brief Constructor
     * @param theta Opening angle parameter (MAC criterion)
     * @param interactionForce Force calculation implementation
     * @param comm MPI communicator (default: MPI_COMM_WORLD)
     */
    MPIBarnesHut(double theta, InteractionForce& interactionForce, MPI_Comm comm = MPI_COMM_WORLD);

    /**
     * @brief Destructor
     */
    ~MPIBarnesHut();

    /**
     * @brief Run the MPI-parallelized Barnes-Hut algorithm
     * 
     * This is the main entry point. Each rank computes forces for its local particles,
     * communicating only the necessary tree data.
     * 
     * @param particles Local particle data (SOA format)
     * @param forceX Output: X component of forces
     * @param forceY Output: Y component of forces
     * @param forceZ Output: Z component of forces
     */
    void run(const ParticleData& particles,
             std::vector<double>& forceX,
             std::vector<double>& forceY,
             std::vector<double>& forceZ);

    /**
     * @brief Get MPI rank
     */
    int getRank() const { return m_rank; }

    /**
     * @brief Get number of MPI processes
     */
    int getNumRanks() const { return m_numRanks; }

    /**
     * @brief Get local particle count for this rank
     */
    size_t getLocalParticleCount() const { return m_localParticleCount; }

    /**
     * @brief Structure to represent a pseudo-particle (aggregate of remote particles)
     */
    struct PseudoParticle {
        std::array<double, 3> centerOfMass;
        double totalMass;
        double size;  // Cube half-width
        
        PseudoParticle() : centerOfMass{0, 0, 0}, totalMass(0), size(0) {}
        PseudoParticle(const std::array<double, 3>& com, double mass, double s)
            : centerOfMass(com), totalMass(mass), size(s) {}
    };

private:
    // MPI configuration
    MPI_Comm m_comm;
    int m_rank;
    int m_numRanks;

    // Algorithm parameters
    double m_theta;
    InteractionForce& m_interactionForce;

    // Domain decomposition data
    std::vector<size_t> m_localIndices;     // Indices of particles owned by this rank
    size_t m_localParticleCount;
    std::array<double, 3> m_globalMinBound;
    std::array<double, 3> m_globalMaxBound;

    // Local Essential Tree data from other ranks
    std::vector<PseudoParticle> m_remotePseudoParticles;

    /**
     * @brief Compute global bounding box across all ranks
     */
    void computeGlobalBounds(const ParticleData& particles);

    /**
     * @brief Partition particles among ranks using Morton curve
     */
    void partitionParticles(const ParticleData& particles);

    /**
     * @brief Build local octree for this rank's particles
     */
    void buildLocalTree(const ParticleData& particles);

    /**
     * @brief Exchange Local Essential Tree data with other ranks
     * 
     * Each rank sends its aggregated tree nodes (pseudo-particles) to ranks
     * that might need them for force calculations.
     */
    void exchangeEssentialTree(const ParticleData& particles);

    /**
     * @brief Calculate forces for local particles
     */
    void calculateLocalForces(const ParticleData& particles,
                              std::vector<double>& forceX,
                              std::vector<double>& forceY,
                              std::vector<double>& forceZ);

    /**
     * @brief Check if a pseudo-particle is sufficiently far for approximation
     */
    bool isSufficientlyFar(const PseudoParticle& pseudo,
                          const std::array<double, 3>& particlePos) const;

    /**
     * @brief Calculate force from a pseudo-particle
     */
    std::array<double, 3> calculatePseudoForce(
        const std::array<double, 3>& position, double mass,
        const PseudoParticle& pseudo);

    /**
     * @brief Aggregate local particles into pseudo-particles for export
     * @param particles Local particle data
     * @param level Octree level for aggregation
     * @return Vector of pseudo-particles
     */
    std::vector<PseudoParticle> aggregateLocalParticles(
        const ParticleData& particles, int level);
};

/**
 * @brief Helper class for MPI initialization/finalization
 * 
 * RAII wrapper for MPI_Init/MPI_Finalize
 */
class MPIEnvironment {
public:
    MPIEnvironment(int* argc, char*** argv) {
        MPI_Init(argc, argv);
    }
    
    ~MPIEnvironment() {
        MPI_Finalize();
    }

    // Non-copyable
    MPIEnvironment(const MPIEnvironment&) = delete;
    MPIEnvironment& operator=(const MPIEnvironment&) = delete;
};

/**
 * @brief Gather all forces to all ranks (for comparison/testing)
 */
void gatherAllForces(const std::vector<double>& localForceX,
                     const std::vector<double>& localForceY,
                     const std::vector<double>& localForceZ,
                     std::vector<double>& globalForceX,
                     std::vector<double>& globalForceY,
                     std::vector<double>& globalForceZ,
                     const std::vector<int>& counts,
                     const std::vector<int>& displacements,
                     MPI_Comm comm);

/**
 * @brief Update particles using computed forces (SOA, MPI-aware)
 */
void updateParticlesMPI(ParticleData& particles,
                        const std::vector<double>& forceX,
                        const std::vector<double>& forceY,
                        const std::vector<double>& forceZ,
                        double timeStep,
                        const std::vector<size_t>& localIndices);

#endif // MPI_BARNES_HUT_H
