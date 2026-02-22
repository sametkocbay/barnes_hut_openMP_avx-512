/**
 * @file mainMPI.cpp
 * @brief MPI-parallelized Barnes-Hut n-body simulation entry point
 * 
 * This program runs the Barnes-Hut algorithm using MPI for distributed
 * parallelization with Morton curve-based domain decomposition (Jülich approach).
 * 
 * Usage: mpirun -np <num_procs> ./barnesHutMPI [options]
 */

#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <chrono>

#include "initialization.h"
#include "bruteForce.h"
#include "Cube.h"
#include "ParticleData.h"
#include "GravitationalForce.h"
#include "MPIBarnesHut.h"
#include "../config/JsonParser.h"

// Print usage information
void printUsageMPI(const char* programName) {
    std::cout << "Usage: mpirun -np <num_procs> " << programName << " [options]\n"
              << "Options:\n"
              << "  -n, --particles <N>   Number of particles (default: from config)\n"
              << "  -t, --theta <θ>       Opening angle parameter (default: 0.5)\n"
              << "  -s, --steps <N>       Number of simulation steps (default: from config)\n"
              << "  -h, --help            Show this help message\n"
              << std::endl;
}

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int rank, numRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    // Parse command line arguments
    int nParticles = 0;
    double theta = 0.5;
    int nSteps = 0;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            if (rank == 0) printUsageMPI(argv[0]);
            MPI_Finalize();
            return 0;
        } else if ((arg == "-n" || arg == "--particles") && i + 1 < argc) {
            nParticles = std::stoi(argv[++i]);
        } else if ((arg == "-t" || arg == "--theta") && i + 1 < argc) {
            theta = std::stod(argv[++i]);
        } else if ((arg == "-s" || arg == "--steps") && i + 1 < argc) {
            nSteps = std::stoi(argv[++i]);
        }
    }

    // Load config (with paths relative to source directory)
    const std::string configFilePath = "../config.json";
    Config config;

    if (rank == 0) {
        std::cout << "=== MPI Barnes-Hut N-Body Simulation ===\n";
        std::cout << "MPI Ranks: " << numRanks << std::endl;
    }

    try {
        config = parseConfig(configFilePath);
    } catch (const std::exception& e) {
        if (rank == 0) {
            std::cerr << "Error loading config: " << e.what() << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    // Override config with command line args
    if (nParticles > 0) config.numParticles = nParticles;
    if (nSteps > 0) config.nSteps = nSteps;

    if (rank == 0) {
        std::cout << "Number of particles: " << config.numParticles << std::endl;
        std::cout << "Simulation steps: " << config.nSteps << std::endl;
        std::cout << "Theta (MAC parameter): " << theta << std::endl;
        std::cout << "Time step: " << config.timeStep << std::endl;
    }

    // Synchronize before starting simulation
    MPI_Barrier(MPI_COMM_WORLD);

    // Generate particles (all ranks generate the same particles for simplicity)
    // In a production code, rank 0 would generate and broadcast
    ParticleData particles = generateRandomParticlesSOA(
        config.numParticles,
        config.massRange,
        config.positionRange,
        config.velocityRange
    );

    // Force vectors
    std::vector<double> forceX, forceY, forceZ;

    // Create MPI Barnes-Hut algorithm instance
    GravitationalForce gforce;
    MPIBarnesHut mpiAlgo(theta, gforce);

    if (rank == 0) {
        std::cout << "\nStarting MPI Barnes-Hut simulation..." << std::endl;
    }

    // Timing
    MPI_Barrier(MPI_COMM_WORLD);
    auto startTime = std::chrono::high_resolution_clock::now();

    // Run simulation
    for (int step = 0; step < config.nSteps; ++step) {
        // Compute forces using MPI Barnes-Hut
        mpiAlgo.run(particles, forceX, forceY, forceZ);

        // Update particles
        updateParticlesSOA(particles, forceX, forceY, forceZ, config.timeStep);

        if (rank == 0 && (step + 1) % 10 == 0) {
            std::cout << "Step " << (step + 1) << "/" << config.nSteps << " completed" << std::endl;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedTime = endTime - startTime;

    if (rank == 0) {
        std::cout << "\nSimulation completed!" << std::endl;
        std::cout << "Total time: " << elapsedTime.count() << " seconds" << std::endl;
        std::cout << "Time per step: " << (elapsedTime.count() / config.nSteps) << " seconds" << std::endl;
    }

    // Report local particle distribution
    size_t localCount = mpiAlgo.getLocalParticleCount();
    if (rank == 0) {
        std::cout << "\nParticle distribution across ranks:" << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    for (int r = 0; r < numRanks; ++r) {
        if (r == rank) {
            std::cout << "  Rank " << rank << ": " << localCount << " particles" << std::endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
