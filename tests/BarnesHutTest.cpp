/**
 * @file BarnesHutTest.cpp
 * @brief Unit tests comparing Barnes-Hut algorithm against Brute Force for n-body simulation
 *
 * Tests run 100 particles for 100 timesteps with dt=0.1 for different theta values.
 * Each theta configuration must remain within a specified error margin to pass.
 * Tests use SOA (Structure of Arrays) layout for better performance.
 */

#include <gtest/gtest.h>
#include <vector>
#include <array>
#include <cmath>
#include <random>

#include "Particle.h"
#include "ParticleData.h"
#include "Cube.h"
#include "initialization.h"
#include "bruteForce.h"
#include "MeshTree.h"
#include "Algorithm.h"
#include "GravitationalForce.h"
#include <omp.h>

namespace {

// Test configuration constants
constexpr int NUM_PARTICLES = 100;
constexpr int NUM_STEPS = 100;
constexpr double TIME_STEP = 0.02;
constexpr unsigned int RANDOM_SEED = 42;  // Fixed seed for reproducibility

/**
 * @brief Generate deterministic random particles for testing (SOA)
 */
ParticleData generateTestParticlesSOA(int numParticles) {
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
 * @brief Run simulation using Brute Force algorithm (SOA)
 */
ParticleData runBruteForceSimulationSOA(ParticleData particles, int nSteps, double dt) {
    std::vector<double> forceX, forceY, forceZ;
    for (int step = 0; step < nSteps; ++step) {
        computeGravitationalForcesSOA(particles, forceX, forceY, forceZ);
        updateParticlesSOA(particles, forceX, forceY, forceZ, dt);
    }
    return particles;
}

/**
 * @brief Run simulation using Barnes-Hut algorithm (SOA)
 */
ParticleData runBarnesHutSimulationSOA(ParticleData particles, int nSteps, double dt, double theta) {
    GravitationalForce gforce;
    std::vector<double> forceX, forceY, forceZ;

    for (int step = 0; step < nSteps; ++step) {
        Cube initialCube = calculateFirstCubeSOA(particles);
        MeshTree tree{initialCube};
        Node* root = tree.divideSpaceSOA(particles);

        Algorithm barnesHutAlgo(theta, gforce);
        barnesHutAlgo.runSOA_SIMD(root, particles, forceX, forceY, forceZ);
        updateParticlesSOA(particles, forceX, forceY, forceZ, dt);
    }
    return particles;
}

/**
 * @brief Calculate average position error between two particle sets (SOA)
 */
double calculateAverageErrorSOA(const ParticleData& particles1,
                                 const ParticleData& particles2) {
    double totalError = 0.0;
    size_t n = particles1.size();

    for (size_t i = 0; i < n; ++i) {
        double dx = particles1.posX[i] - particles2.posX[i];
        double dy = particles1.posY[i] - particles2.posY[i];
        double dz = particles1.posZ[i] - particles2.posZ[i];

        totalError += std::sqrt(dx*dx + dy*dy + dz*dz);
    }

    return totalError / n;
}

/**
 * @brief Calculate maximum position error between two particle sets (SOA)
 */
double calculateMaxErrorSOA(const ParticleData& particles1,
                             const ParticleData& particles2) {
    double maxError = 0.0;
    size_t n = particles1.size();

    for (size_t i = 0; i < n; ++i) {
        double dx = particles1.posX[i] - particles2.posX[i];
        double dy = particles1.posY[i] - particles2.posY[i];
        double dz = particles1.posZ[i] - particles2.posZ[i];

        double error = std::sqrt(dx*dx + dy*dy + dz*dz);
        maxError = std::max(maxError, error);
    }

    return maxError;
}



} // anonymous namespace


// ============================================================================
// SOA (Structure of Arrays) Tests
// ============================================================================

/**
 * @brief Test fixture for Barnes-Hut vs Brute Force comparison (SOA)
 */
class BarnesHutAccuracyTestSOA : public ::testing::Test {
protected:
    ParticleData initialParticles;
    ParticleData bruteForceResult;

    void SetUp() override {
        initialParticles = generateTestParticlesSOA(NUM_PARTICLES);
        bruteForceResult = runBruteForceSimulationSOA(initialParticles, NUM_STEPS, TIME_STEP);
    }
};


/**
 * @brief Test Barnes-Hut SOA with theta = 0.0 (should be equivalent to brute force)
 */
TEST_F(BarnesHutAccuracyTestSOA, Theta0_ExactMatch) {
    const double theta = 0.0;
    const double maxAllowedError = 1e-10;

    ParticleData barnesHutResult = runBarnesHutSimulationSOA(initialParticles, NUM_STEPS, TIME_STEP, theta);

    double avgError = calculateAverageErrorSOA(bruteForceResult, barnesHutResult);
    double maxError = calculateMaxErrorSOA(bruteForceResult, barnesHutResult);

    std::cout << "             " << "[SOA θ=0.0] Average Error: " << avgError << ", Max Error: " << maxError << std::endl;

    EXPECT_LT(avgError, maxAllowedError)
        << "SOA: Theta=0 should produce results nearly identical to brute force";
    EXPECT_LT(maxError, maxAllowedError)
        << "SOA: Maximum error exceeded for theta=0";
}


/**
 * @brief Test Barnes-Hut SOA with theta = 0.3 (high accuracy)
 */
TEST_F(BarnesHutAccuracyTestSOA, Theta03_HighAccuracy) {
    const double theta = 0.3;
    const double maxAllowedAvgError = 5.0;
    const double maxAllowedMaxError = 20.0;

    ParticleData barnesHutResult = runBarnesHutSimulationSOA(initialParticles, NUM_STEPS, TIME_STEP, theta);

    double avgError = calculateAverageErrorSOA(bruteForceResult, barnesHutResult);
    double maxError = calculateMaxErrorSOA(bruteForceResult, barnesHutResult);

    std::cout << "             " << "[SOA θ=0.3] Average Error: " << avgError << ", Max Error: " << maxError << std::endl;

    EXPECT_LT(avgError, maxAllowedAvgError)
        << "SOA: Average error exceeded for theta=0.3";
    EXPECT_LT(maxError, maxAllowedMaxError)
        << "SOA: Maximum error exceeded for theta=0.3";
}


/**
 * @brief Test Barnes-Hut SOA with theta = 0.5 (balanced accuracy/performance)
 */
TEST_F(BarnesHutAccuracyTestSOA, Theta05_BalancedAccuracy) {
    const double theta = 0.5;
    const double maxAllowedAvgError = 10.0;
    const double maxAllowedMaxError = 50.0;

    ParticleData barnesHutResult = runBarnesHutSimulationSOA(initialParticles, NUM_STEPS, TIME_STEP, theta);

    double avgError = calculateAverageErrorSOA(bruteForceResult, barnesHutResult);
    double maxError = calculateMaxErrorSOA(bruteForceResult, barnesHutResult);

    std::cout << "             " << "[SOA θ=0.5] Average Error: " << avgError << ", Max Error: " << maxError << std::endl;

    EXPECT_LT(avgError, maxAllowedAvgError)
        << "SOA: Average error exceeded for theta=0.5";
    EXPECT_LT(maxError, maxAllowedMaxError)
        << "SOA: Maximum error exceeded for theta=0.5";
}


/**
 * @brief Test Barnes-Hut SOA with theta = 0.7 (faster but less accurate)
 */
TEST_F(BarnesHutAccuracyTestSOA, Theta07_LowerAccuracy) {
    const double theta = 0.7;
    const double maxAllowedAvgError = 25.0;
    const double maxAllowedMaxError = 100.0;

    ParticleData barnesHutResult = runBarnesHutSimulationSOA(initialParticles, NUM_STEPS, TIME_STEP, theta);

    double avgError = calculateAverageErrorSOA(bruteForceResult, barnesHutResult);
    double maxError = calculateMaxErrorSOA(bruteForceResult, barnesHutResult);

    std::cout << "             " << "[SOA θ=0.7] Average Error: " << avgError << ", Max Error: " << maxError << std::endl;

    EXPECT_LT(avgError, maxAllowedAvgError)
        << "SOA: Average error exceeded for theta=0.7";
    EXPECT_LT(maxError, maxAllowedMaxError)
        << "SOA: Maximum error exceeded for theta=0.7";
}


/**
 * @brief Test Barnes-Hut SOA with theta = 1.0 (maximum speed, lowest accuracy)
 */
TEST_F(BarnesHutAccuracyTestSOA, Theta10_LowestAccuracy) {
    const double theta = 1.0;
    const double maxAllowedAvgError = 50.0;
    const double maxAllowedMaxError = 200.0;

    ParticleData barnesHutResult = runBarnesHutSimulationSOA(initialParticles, NUM_STEPS, TIME_STEP, theta);

    double avgError = calculateAverageErrorSOA(bruteForceResult, barnesHutResult);
    double maxError = calculateMaxErrorSOA(bruteForceResult, barnesHutResult);

    std::cout << "             " << "[SOA θ=1.0] Average Error: " << avgError << ", Max Error: " << maxError << std::endl;

    EXPECT_LT(avgError, maxAllowedAvgError)
        << "SOA: Average error exceeded for theta=1.0";
    EXPECT_LT(maxError, maxAllowedMaxError)
        << "SOA: Maximum error exceeded for theta=1.0";
}


/**
 * @brief Test that SOA error increases monotonically with theta
 */
TEST_F(BarnesHutAccuracyTestSOA, ErrorIncreasesWithTheta) {
    std::vector<double> thetas = {0.3, 0.5, 0.7, 1.0};
    std::vector<double> errors;

    for (double theta : thetas) {
        ParticleData barnesHutResult = runBarnesHutSimulationSOA(initialParticles, NUM_STEPS, TIME_STEP, theta);
        errors.push_back(calculateAverageErrorSOA(bruteForceResult, barnesHutResult));
    }

    for (size_t i = 1; i < errors.size(); ++i) {
        std::cout << "             "
                  << "[SOA] Error for θ=" << thetas[i-1] << ": " << errors[i-1]
                  << " vs theta=" << thetas[i] << ": " << errors[i] << std::endl;
    }

    EXPECT_GT(errors.back(), errors.front())
        << "SOA: Error should generally increase with larger theta values";
}



/**
 * @brief Main function to run all tests
 */
int main(int argc, char **argv) {
    #pragma omp parallel
    #pragma omp single
    {
        std::cout << "\n\nRunning tests with " << omp_get_num_threads() << " OpenMP threads.\n\n" << std::endl;
    }
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
