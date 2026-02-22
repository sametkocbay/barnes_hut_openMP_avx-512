/**
 * @file MPIBarnesHutTest.cpp
 * @brief Comparison tests for MPI Barnes-Hut vs Brute Force vs OpenMP+SIMD Barnes-Hut
 * 
 * This test suite compares the accuracy and performance of three implementations:
 * 1. Brute Force (O(N²)) - Reference implementation
 * 2. OpenMP+SIMD Barnes-Hut - Existing parallel implementation
 * 3. MPI Barnes-Hut (Jülich approach) - New distributed implementation
 * 
 * Tests are run with varying numbers of particles and theta values.
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <vector>
#include <array>
#include <cmath>
#include <random>
#include <chrono>
#include <iostream>
#include <iomanip>

#include "Particle.h"
#include "ParticleData.h"
#include "Cube.h"
#include "initialization.h"
#include "bruteForce.h"
#include "MeshTree.h"
#include "Algorithm.h"
#include "GravitationalForce.h"
#include "MPIBarnesHut.h"
#include "MortonCurve.h"

namespace {

// Test configuration
constexpr int NUM_PARTICLES = 100;
constexpr int NUM_STEPS = 10;
constexpr double TIME_STEP = 0.02;
constexpr unsigned int RANDOM_SEED = 42;

// MPI rank info (set in main)
int g_rank = 0;
int g_numRanks = 1;

/**
 * @brief Generate deterministic random particles
 */
ParticleData generateTestParticles(int numParticles) {
    ParticleData particles;
    particles.reserve(numParticles);
    std::mt19937 gen(RANDOM_SEED);
    std::uniform_real_distribution<> posDist(-10.0, 10.0);
    std::uniform_real_distribution<> velDist(-1.0, 1.0);
    std::uniform_real_distribution<> massDist(0.1, 2.0);

    for (int i = 0; i < numParticles; ++i) {
        particles.addParticle(
            posDist(gen), posDist(gen), posDist(gen),
            velDist(gen), velDist(gen), velDist(gen),
            massDist(gen)
        );
    }
    return particles;
}

/**
 * @brief Run Brute Force simulation
 */
ParticleData runBruteForce(ParticleData particles, int nSteps, double dt) {
    std::vector<double> forceX, forceY, forceZ;
    for (int step = 0; step < nSteps; ++step) {
        computeGravitationalForcesSOA(particles, forceX, forceY, forceZ);
        updateParticlesSOA(particles, forceX, forceY, forceZ, dt);
    }
    return particles;
}

/**
 * @brief Run OpenMP+SIMD Barnes-Hut simulation
 */
ParticleData runOpenMPBarnesHut(ParticleData particles, int nSteps, double dt, double theta) {
    GravitationalForce gforce;
    std::vector<double> forceX, forceY, forceZ;

    for (int step = 0; step < nSteps; ++step) {
        Cube initialCube = calculateFirstCubeSOA(particles);
        MeshTree tree{initialCube};
        Node* root = tree.divideSpaceSOA(particles);

        Algorithm algo(theta, gforce);
        algo.runSOA_SIMD(root, particles, forceX, forceY, forceZ);
        updateParticlesSOA(particles, forceX, forceY, forceZ, dt);
    }
    return particles;
}

/**
 * @brief Run MPI Barnes-Hut simulation
 */
ParticleData runMPIBarnesHut(ParticleData particles, int nSteps, double dt, double theta) {
    GravitationalForce gforce;
    MPIBarnesHut mpiAlgo(theta, gforce);
    std::vector<double> forceX, forceY, forceZ;

    for (int step = 0; step < nSteps; ++step) {
        mpiAlgo.run(particles, forceX, forceY, forceZ);
        updateParticlesSOA(particles, forceX, forceY, forceZ, dt);
    }
    return particles;
}

/**
 * @brief Calculate average position error between particle sets
 */
double calculateAverageError(const ParticleData& p1, const ParticleData& p2) {
    double totalError = 0.0;
    size_t n = p1.size();
    for (size_t i = 0; i < n; ++i) {
        double dx = p1.posX[i] - p2.posX[i];
        double dy = p1.posY[i] - p2.posY[i];
        double dz = p1.posZ[i] - p2.posZ[i];
        totalError += std::sqrt(dx*dx + dy*dy + dz*dz);
    }
    return totalError / n;
}

/**
 * @brief Calculate maximum position error
 */
double calculateMaxError(const ParticleData& p1, const ParticleData& p2) {
    double maxError = 0.0;
    size_t n = p1.size();
    for (size_t i = 0; i < n; ++i) {
        double dx = p1.posX[i] - p2.posX[i];
        double dy = p1.posY[i] - p2.posY[i];
        double dz = p1.posZ[i] - p2.posZ[i];
        double error = std::sqrt(dx*dx + dy*dy + dz*dz);
        maxError = std::max(maxError, error);
    }
    return maxError;
}

} // anonymous namespace

// ============================================================================
// Morton Curve Unit Tests
// ============================================================================

class MortonCurveTest : public ::testing::Test {};

TEST_F(MortonCurveTest, EncodeDecodeRoundTrip) {
    // Test that encode-decode is reversible
    std::vector<std::array<uint32_t, 3>> testCoords = {
        {0, 0, 0},
        {1, 1, 1},
        {100, 200, 300},
        {1000, 2000, 3000},
        {(1 << 20) - 1, (1 << 20) - 1, (1 << 20) - 1}  // Max 21-bit values
    };

    for (const auto& coord : testCoords) {
        uint64_t code = MortonCurve::encode(coord[0], coord[1], coord[2]);
        auto decoded = MortonCurve::decode(code);
        
        EXPECT_EQ(decoded[0], coord[0]) << "X mismatch for (" << coord[0] << "," << coord[1] << "," << coord[2] << ")";
        EXPECT_EQ(decoded[1], coord[1]) << "Y mismatch";
        EXPECT_EQ(decoded[2], coord[2]) << "Z mismatch";
    }
}

TEST_F(MortonCurveTest, NormalizedEncoding) {
    // Test that normalized coordinates within [0,1] produce valid codes
    uint64_t code000 = MortonCurve::encodeNormalized(0.0, 0.0, 0.0);
    uint64_t code111 = MortonCurve::encodeNormalized(1.0, 1.0, 1.0);
    uint64_t code050 = MortonCurve::encodeNormalized(0.5, 0.5, 0.5);

    EXPECT_LT(code000, code050);
    EXPECT_LT(code050, code111);
}

TEST_F(MortonCurveTest, SpatialLocality) {
    // Test that nearby points have similar Morton codes
    uint64_t code1 = MortonCurve::encodeNormalized(0.1, 0.1, 0.1);
    uint64_t code2 = MortonCurve::encodeNormalized(0.11, 0.11, 0.11);
    uint64_t code3 = MortonCurve::encodeNormalized(0.9, 0.9, 0.9);

    // Difference between nearby points should be smaller than distant points
    uint64_t diff_nearby = (code1 > code2) ? (code1 - code2) : (code2 - code1);
    uint64_t diff_distant = (code1 > code3) ? (code1 - code3) : (code3 - code1);

    // Not a strict requirement, but generally holds for Morton curves
    EXPECT_LT(diff_nearby, diff_distant);
}

TEST_F(MortonCurveTest, ParticleSorting) {
    // Test particle sorting by Morton code
    std::vector<double> posX = {0.1, 0.9, 0.5, 0.2, 0.8};
    std::vector<double> posY = {0.1, 0.9, 0.5, 0.2, 0.8};
    std::vector<double> posZ = {0.1, 0.9, 0.5, 0.2, 0.8};
    std::array<double, 3> minBound = {0, 0, 0};
    std::array<double, 3> maxBound = {1, 1, 1};

    auto sorted = MortonCurve::sortParticlesByMorton(posX, posY, posZ, minBound, maxBound);

    EXPECT_EQ(sorted.size(), 5);
    
    // Check that codes are in non-decreasing order
    for (size_t i = 1; i < sorted.size(); ++i) {
        EXPECT_LE(sorted[i-1].first, sorted[i].first);
    }
}

// ============================================================================
// MPI Barnes-Hut Accuracy Tests
// ============================================================================

class MPIBarnesHutAccuracyTest : public ::testing::Test {
protected:
    ParticleData initialParticles;
    ParticleData bruteForceResult;

    void SetUp() override {
        initialParticles = generateTestParticles(NUM_PARTICLES);
        bruteForceResult = runBruteForce(initialParticles, NUM_STEPS, TIME_STEP);
    }
};

TEST_F(MPIBarnesHutAccuracyTest, MPIvsBruteForce_Theta0p5) {
    // Test MPI Barnes-Hut with theta = 0.5
    double theta = 0.5;
    ParticleData mpiResult = runMPIBarnesHut(initialParticles, NUM_STEPS, TIME_STEP, theta);

    double avgError = calculateAverageError(bruteForceResult, mpiResult);
    double maxError = calculateMaxError(bruteForceResult, mpiResult);

    if (g_rank == 0) {
        std::cout << "\n[MPI vs Brute Force, theta=" << theta << "]\n"
                  << "  Average error: " << avgError << "\n"
                  << "  Maximum error: " << maxError << std::endl;
    }

    // Error thresholds (MPI with approximation will have some error)
    EXPECT_LT(avgError, 1.0) << "Average error too large for theta=" << theta;
    EXPECT_LT(maxError, 5.0) << "Max error too large";
}

TEST_F(MPIBarnesHutAccuracyTest, OpenMPvsBruteForce_Theta0p5) {
    // Test OpenMP Barnes-Hut for comparison
    double theta = 0.5;
    ParticleData ompResult = runOpenMPBarnesHut(initialParticles, NUM_STEPS, TIME_STEP, theta);

    double avgError = calculateAverageError(bruteForceResult, ompResult);
    double maxError = calculateMaxError(bruteForceResult, ompResult);

    if (g_rank == 0) {
        std::cout << "\n[OpenMP vs Brute Force, theta=" << theta << "]\n"
                  << "  Average error: " << avgError << "\n"
                  << "  Maximum error: " << maxError << std::endl;
    }

    EXPECT_LT(avgError, 0.5) << "OpenMP average error too large";
    EXPECT_LT(maxError, 2.0) << "OpenMP max error too large";
}

TEST_F(MPIBarnesHutAccuracyTest, MPIvsOpenMP_Theta0p5) {
    // Compare MPI and OpenMP implementations directly
    double theta = 0.5;
    ParticleData mpiResult = runMPIBarnesHut(initialParticles, NUM_STEPS, TIME_STEP, theta);
    ParticleData ompResult = runOpenMPBarnesHut(initialParticles, NUM_STEPS, TIME_STEP, theta);

    double avgError = calculateAverageError(mpiResult, ompResult);
    double maxError = calculateMaxError(mpiResult, ompResult);

    if (g_rank == 0) {
        std::cout << "\n[MPI vs OpenMP, theta=" << theta << "]\n"
                  << "  Average error: " << avgError << "\n"
                  << "  Maximum error: " << maxError << std::endl;
    }

    // MPI and OpenMP should be reasonably close (both use approximations)
    EXPECT_LT(avgError, 1.5) << "MPI and OpenMP diverged too much";
}

// ============================================================================
// Performance Comparison Tests
// ============================================================================

class PerformanceComparisonTest : public ::testing::Test {};

TEST_F(PerformanceComparisonTest, CompareAllImplementations) {
    // Run and time all three implementations
    const int numParticles = 500;
    const int numSteps = 5;
    const double theta = 0.5;

    ParticleData particles = generateTestParticles(numParticles);

    // Brute Force timing
    MPI_Barrier(MPI_COMM_WORLD);
    auto t1 = std::chrono::high_resolution_clock::now();
    ParticleData bf_result = runBruteForce(particles, numSteps, TIME_STEP);
    auto t2 = std::chrono::high_resolution_clock::now();
    double bf_time = std::chrono::duration<double>(t2 - t1).count();

    // OpenMP Barnes-Hut timing
    MPI_Barrier(MPI_COMM_WORLD);
    t1 = std::chrono::high_resolution_clock::now();
    ParticleData omp_result = runOpenMPBarnesHut(particles, numSteps, TIME_STEP, theta);
    t2 = std::chrono::high_resolution_clock::now();
    double omp_time = std::chrono::duration<double>(t2 - t1).count();

    // MPI Barnes-Hut timing
    MPI_Barrier(MPI_COMM_WORLD);
    t1 = std::chrono::high_resolution_clock::now();
    ParticleData mpi_result = runMPIBarnesHut(particles, numSteps, TIME_STEP, theta);
    t2 = std::chrono::high_resolution_clock::now();
    double mpi_time = std::chrono::duration<double>(t2 - t1).count();

    // Calculate errors
    double omp_error = calculateAverageError(bf_result, omp_result);
    double mpi_error = calculateAverageError(bf_result, mpi_result);

    if (g_rank == 0) {
        std::cout << "\n" << std::string(60, '=') << "\n"
                  << "Performance Comparison (" << numParticles << " particles, "
                  << numSteps << " steps, " << g_numRanks << " MPI ranks)\n"
                  << std::string(60, '=') << "\n"
                  << std::fixed << std::setprecision(4)
                  << "| Implementation      | Time (s)  | Avg Error | Speedup   |\n"
                  << "|---------------------|-----------|-----------|----------|\n"
                  << "| Brute Force         | " << std::setw(9) << bf_time
                  << " | " << std::setw(9) << 0.0 << " | " << std::setw(8) << 1.0 << "x |\n"
                  << "| OpenMP Barnes-Hut   | " << std::setw(9) << omp_time
                  << " | " << std::setw(9) << omp_error << " | " << std::setw(8) << (bf_time/omp_time) << "x |\n"
                  << "| MPI Barnes-Hut      | " << std::setw(9) << mpi_time
                  << " | " << std::setw(9) << mpi_error << " | " << std::setw(8) << (bf_time/mpi_time) << "x |\n"
                  << std::string(60, '=') << std::endl;
    }

    // Basic sanity check - errors should be bounded
    EXPECT_LT(omp_error, 1.0);
    EXPECT_LT(mpi_error, 2.0);
}

// ============================================================================
// MPI-specific Tests
// ============================================================================

class MPISpecificTest : public ::testing::Test {};

TEST_F(MPISpecificTest, DomainPartitioning) {
    // Test that domain partitioning works correctly
    ParticleData particles = generateTestParticles(NUM_PARTICLES);

    GravitationalForce gforce;
    MPIBarnesHut mpiAlgo(0.5, gforce);

    std::vector<double> fx, fy, fz;
    mpiAlgo.run(particles, fx, fy, fz);

    // Check that local particle count is reasonable
    size_t localCount = mpiAlgo.getLocalParticleCount();
    size_t expectedCount = NUM_PARTICLES / g_numRanks;

    // Allow ±1 for rounding
    EXPECT_GE(localCount, expectedCount - 1);
    EXPECT_LE(localCount, expectedCount + 1);

    // Gather all counts and verify total
    std::vector<size_t> allCounts(g_numRanks);
    size_t myCount = localCount;
    MPI_Allgather(&myCount, 1, MPI_UNSIGNED_LONG,
                  allCounts.data(), 1, MPI_UNSIGNED_LONG, MPI_COMM_WORLD);

    size_t totalCount = 0;
    for (int i = 0; i < g_numRanks; ++i) {
        totalCount += allCounts[i];
    }

    EXPECT_EQ(totalCount, NUM_PARTICLES);

    if (g_rank == 0) {
        std::cout << "\nDomain partition distribution:\n";
        for (int i = 0; i < g_numRanks; ++i) {
            std::cout << "  Rank " << i << ": " << allCounts[i] << " particles\n";
        }
    }
}

TEST_F(MPISpecificTest, ForceConsistency) {
    // Test that all ranks compute consistent forces
    ParticleData particles = generateTestParticles(NUM_PARTICLES);

    GravitationalForce gforce;
    MPIBarnesHut mpiAlgo(0.5, gforce);

    std::vector<double> fx, fy, fz;
    mpiAlgo.run(particles, fx, fy, fz);

    // Forces should be synchronized across ranks (due to Allreduce)
    // Spot check a few force values
    std::vector<double> rank0_forces(3);
    if (g_rank == 0 && particles.size() > 0) {
        rank0_forces[0] = fx[0];
        rank0_forces[1] = fy[0];
        rank0_forces[2] = fz[0];
    }
    MPI_Bcast(rank0_forces.data(), 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (particles.size() > 0) {
        EXPECT_DOUBLE_EQ(fx[0], rank0_forces[0]) << "Force X mismatch across ranks";
        EXPECT_DOUBLE_EQ(fy[0], rank0_forces[1]) << "Force Y mismatch across ranks";
        EXPECT_DOUBLE_EQ(fz[0], rank0_forces[2]) << "Force Z mismatch across ranks";
    }
}

// ============================================================================
// Main with MPI initialization
// ============================================================================

int main(int argc, char** argv) {
    // Initialize MPI before GTest
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_numRanks);

    // Initialize GTest
    ::testing::InitGoogleTest(&argc, argv);

    // Only rank 0 prints test output
    if (g_rank != 0) {
        ::testing::TestEventListeners& listeners =
            ::testing::UnitTest::GetInstance()->listeners();
        delete listeners.Release(listeners.default_result_printer());
    }

    if (g_rank == 0) {
        std::cout << "\n========================================\n"
                  << "MPI Barnes-Hut Comparison Tests\n"
                  << "Running with " << g_numRanks << " MPI ranks\n"
                  << "========================================\n" << std::endl;
    }

    int result = RUN_ALL_TESTS();

    MPI_Finalize();
    return result;
}
