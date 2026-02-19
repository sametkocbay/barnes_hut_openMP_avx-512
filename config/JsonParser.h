#ifndef BARNESHUT_JSONPARSER_H
#define BARNESHUT_JSONPARSER_H

#include <string>
#include <vector>
#include <array>
#include "Particle.h"

// Struct that holds the configuration parameters parsed from the config.json file
struct Config {
    std::array<double, 2> massRange;      // Range of particle masses [min, max] in [kg]
    std::array<double, 2> positionRange;  // Range of particle positions [min, max] in [m]
    std::array<double, 2> velocityRange;  // Range of particle velocities [min, max] in [m/s]
    double timeStep;                      // Time step for the simulation: [s] timestep dt
    double maxTime;                       // Maximum time for the simulation: [s] full time duration to calc
    int nSteps;                           // Number of simulation time steps (maxTime / timeStep)
    int numParticles;                     // Number of particles
    bool barnesHutAlgorithm;              // Flag indicating whether to use Barnes-Hut algorithm instead of Brute-Force-Algorithm
    double unittestTolerance;             // Tolerance value for unit tests
    double MAC;                           // Constant value defining the distance between cluster and particle as fraction of clusterWidth and distance
    int treeParallelLevel;                // Octree parallel construction level: 1=8, 2=64, 3=512, 4=4096 subtrees
};

// Parses all Parameters of the configuration file in JSON format and returns a Config struct.
// The function reads the file, extracts parameters, and initializes the fields of the given Config struct.
// Throws runtime_error if parsing fails or a required key is not found.
Config parseConfig(const std::string& configFilePath);

// Reads the content of a file into a string.
// The file is opened at the given file path and its contents are read into a string buffer.
// Throws runtime_error if the file cannot be opened.
std::string readFile(const std::string &filePath);

// These functions parse the according parameter type from a JSON string based on a given key.
// Searches the JSON content for the key, extracts the parameter, and returns them as the required parameter type.
// Throws runtime_error if the key is not found or the parameter is malformed.
std::vector<double> parseArray(const std::string &key, const std::string &json);
double parseDouble(const std::string &key, const std::string &json);
bool parseBoolean(const std::string &key, const std::string &json);

// Parse a scenario from scenarios.json and return particles
std::vector<Particle> parseScenario(const std::string& scenariosFilePath, const std::string& scenarioName);

// Get list of available scenario names
std::vector<std::string> getAvailableScenarios(const std::string& scenariosFilePath);

// Print usage information
void printUsage(const std::string& programName);

#endif //BARNESHUT_JSONPARSER_H
