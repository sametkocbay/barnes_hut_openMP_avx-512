#include "initialization.h"
#include "bruteForce.h"
#include <iostream>
#include <fstream>
#include "MeshTree.h"
#include <vector>
#include <array>
#include <memory>
#include <Cube.h>
#include <filesystem>
#include "particleNode.h"
#include "Algorithm.h"
#include "../config/JsonParser.h"
#include "ParticleData.h"
#include "InteractionForce.h"
#include <cstring>
#include <chrono>
#include <omp.h>


int main(int argc, char **argv) {

    ///Parse Command Line Arguments
    bool useRandom = true;
    std::string scenarioName;
    int nParticles = 0;
    int treeLevel = 0;  // 0 means use default from config

    // Use source directory paths (relative to project root, not build directory)
    const std::string scenariosFilePath = "../config/scenarios.json";
    const std::string configFilePath    = "../config.json";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        }
        else if (arg == "--list" || arg == "-l") {
            std::cout << "\nAvailable scenarios:\n";
            std::vector<std::string> scenarios = getAvailableScenarios(scenariosFilePath);
            for (const auto& s : scenarios) {
                std::cout << "  - " << s << std::endl;
            }
            std::cout << std::endl;
            return 0;
        }
        else if (arg == "--random" || arg == "-r") {
            useRandom = true;
        }
        else if (arg == "--scenario" || arg == "-s") {
            if (i + 1 < argc) {
                scenarioName = argv[++i];
                useRandom = false;
            } else {
                std::cerr << "Error: --scenario requires a scenario name.\n";
                printUsage(argv[0]);
                return 1;
            }
        }
        else if (arg == "-n") {
            if (i + 1 < argc) {
                nParticles = std::stoi(argv[++i]);
            } else {
                std::cerr << "Error: -n requires number of particles.\n";
                printUsage(argv[0]);
                return 1;
            }
        }
        else if (arg == "--tree-level") {
            if (i + 1 < argc) {
                treeLevel = std::stoi(argv[++i]);
                if (treeLevel < 1 || treeLevel > 4) {
                    std::cerr << "Error: --tree-level must be between 1 and 4.\n";
                    std::cerr << "  1 = 8 subtrees, 2 = 64 subtrees, 3 = 512 subtrees, 4 = 4096 subtrees\n";
                    return 1;
                }
            } else {
                std::cerr << "Error: --tree-level requires a level (1-4).\n";
                printUsage(argv[0]);
                return 1;
            }
        }
        else if (arg == "--simd-batches") {
            if (i + 1 < argc) {
                size_t batches = std::stoul(argv[++i]);
                setSIMDBatches(batches);
                std::cout << "SIMD batches set to: " << SIMD_BATCHES << " (width: " << SIMD_WIDTH << ")\n";
            } else {
                std::cerr << "Error: --simd-batches requires a number (1-16).\n";
                printUsage(argv[0]);
                return 1;
            }
        }
        else {
            std::cerr << "Unknown option: " << arg << std::endl;
            printUsage(argv[0]);
            return 1;
        }
    }

    // Declare Input Parameter Variables (as Struct config)
    Config config;

    ///Debug Check: Check working directory from where to find json
    std::cout << "Working directory: " << std::filesystem::current_path() << std::endl;
    std::cout << "Loading config from: " << configFilePath << std::endl;

    // Get Input parameters from Config File into Config Struct
    try {
        config = parseConfig(configFilePath);
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\n"<< std::endl;
        return 1;
    }

    if (nParticles) {
        config.numParticles = nParticles;
        std::cout << "Number of Particles: " << config.numParticles << "" << std::endl;
    }

    if (treeLevel) {
        config.treeParallelLevel = treeLevel;
    } else {
        config.treeParallelLevel = 1; // Default tree level if not specified
    }
    
    int numSubtrees = 1;
    for (int i = 0; i < config.treeParallelLevel; ++i) numSubtrees *= 8;
    std::cout << "Tree Parallel Level: " << config.treeParallelLevel
              << " (" << numSubtrees << " subtrees)" << std::endl;

    // Container to store particle positions at each time step
    std::vector<std::vector<std::array<double, 3>>> particlePositions(config.nSteps + 1);

    #pragma omp parallel
    #pragma omp single
    {
        std::cout << "Using " << omp_get_num_threads() << " threads for SOA force calculation." << std::endl;
    }

    auto startTime = std::chrono::high_resolution_clock::now();

    ///Create Particles using SOA layout
    ParticleData particles;

    if (useRandom) {
        std::cout << "\nUsing random particles (SOA layout)...\n" << std::endl;
        particles = generateRandomParticlesSOA(config.numParticles, config.massRange, config.positionRange, config.velocityRange);
    } else {
        std::cout << "\nLoading scenario: " << scenarioName << " (converting to SOA)...\n" << std::endl;
        try {
            std::vector<Particle> aosParticles = parseScenario(scenariosFilePath, scenarioName);
            particles = convertToSOA(aosParticles);
        } catch (const std::exception &e) {
            std::cerr << "Error loading scenario: " << e.what() << "\n";
            std::cerr << "Use --list to see available scenarios.\n" << std::endl;
            return 1;
        }
    }

    // Initialize with initial positions
    for (size_t i = 0; i < particles.size(); ++i) {
        particlePositions[0].push_back({particles.posX[i], particles.posY[i], particles.posZ[i]});
    }

    ///Start Simulation
    std::cout << "Starting simulation with " << particles.size() << " particles for "
              << config.nSteps << " steps..." << std::endl;

    // Force vectors for SOA
    std::vector<double> forceX, forceY, forceZ;

    // Run the simulation for nSteps
    for (int step = 0; step < config.nSteps; ++step) {

        if (config.barnesHutAlgorithm) {
            // define the initial cube using SOA

            ///////////////////// Timing Octree Creation /////////////////////
            auto start_tree_creation = std::chrono::high_resolution_clock::now();
            Cube initialCube = calculateFirstCubeSOA(particles);

            // Divide the space based on a space dividing function (here: Octree)
            MeshTree* tree = new MeshTree(initialCube, config.treeParallelLevel);
            Node *root = tree->divideSpaceSOA(particles);
            auto end_tree_creation = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_tree_creation = end_tree_creation - start_tree_creation;
            std::cout << "Step " << step << ": Octree Creation Time: " << elapsed_tree_creation.count() << " seconds" << std::endl;
            ///////////////////// End Timing Octree Creation /////////////////////

            ///////////////////// Timing BH Force Calculation /////////////////////
            auto start_bh_force = std::chrono::high_resolution_clock::now();
            GravitationalForce gforce;
            Algorithm barnesHutAlgo(config.MAC, gforce);
            barnesHutAlgo.runSOA_SIMD(root, particles, forceX, forceY, forceZ);
            auto end_bh_force = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_bh_force = end_bh_force - start_bh_force;
            std::cout << "Step " << step << ": Barnes-Hut Force Calculation Time: " << elapsed_bh_force.count() << " seconds" << std::endl;
            ///////////////////// End Timing BH Force Calculation /////////////////////

            ///////////////////// Timing Octree Destruction /////////////////////
            auto start_tree_destruction = std::chrono::high_resolution_clock::now();
            delete tree;
            auto end_tree_destruction = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_tree_destruction = end_tree_destruction - start_tree_destruction;
            std::cout << "Step " << step << ": Octree Destruction Time: " << elapsed_tree_destruction.count() << " seconds" << std::endl;
            ///////////////////// End Timing Octree Destruction /////////////////////
        }
        else {
            // calculate the forces with the brute force approach
            computeGravitationalForcesSOA(particles, forceX, forceY, forceZ);
        }

        // Update particles using SOA
        ///////////////////// Timing Particle Update /////////////////////
        auto start_update = std::chrono::high_resolution_clock::now();
        updateParticlesSOA(particles, forceX, forceY, forceZ, config.timeStep);
        auto end_update = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_update = end_update - start_update;
        std::cout << "Step " << step << ": Particle Update Time: " << elapsed_update.count() << " seconds" << std::endl;
        ///////////////////// End Timing Particle Update /////////////////////

        // Store the updated positions
        auto update_start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < particles.size(); ++i) {
            particlePositions[step + 1].push_back({particles.posX[i], particles.posY[i], particles.posZ[i]});
        }
        auto update_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_update_store = update_end - update_start;
        std::cout << "Step " << step << ": Position Storage Time: " << elapsed_update_store.count() << " seconds" << std::endl;
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedSeconds = endTime - startTime;
    std::cout << "Simulation Time: " << elapsedSeconds.count() << " seconds\n" << std::endl;

    /// Write results to CSV
    //std::string outputFilename = "../particle_positions.csv";
    //writeToCSV(particlePositions, outputFilename);
    //std::cout << "Simulation complete. Results written to " << outputFilename << std::endl;

    return 0;
}
