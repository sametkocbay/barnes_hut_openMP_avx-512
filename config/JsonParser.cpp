#include "JsonParser.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <iostream>

//Parse everything and return config struct
Config parseConfig(const std::string& configFilePath) {
    Config config;

    try {
        // Read the JSON file
        std::string jsonContent = readFile(configFilePath);
        std::cout << "Successfully read in Config file \n"<< std::endl;

        // Arrays for particle initialization ranges
        config.massRange = {
                parseArray("massRange", jsonContent)[0],
                parseArray("massRange", jsonContent)[1]};
        config.positionRange = {
                parseArray("positionRange", jsonContent)[0],
                parseArray("positionRange", jsonContent)[1]};
        config.velocityRange = {
                parseArray("velocityRange", jsonContent)[0],
                parseArray("velocityRange", jsonContent)[1]};

        // Simulation parameters
        config.timeStep = parseDouble("timeStep", jsonContent);
        config.maxTime = parseDouble("maxTime", jsonContent);
        config.numParticles = static_cast<int>(parseDouble("numParticles", jsonContent));

        config.nSteps = config.maxTime / config.timeStep;

        // Set bools for simulation type
        config.barnesHutAlgorithm = parseBoolean("barnesHutAlgorithm", jsonContent);

        //Set constant values
        config.unittestTolerance = parseDouble("unittestTolerance", jsonContent);
        config.MAC = parseDouble("MAC", jsonContent);

        // Print parsed values for verification (optional)
        std::cout << "Config successfully loaded:\n";
        std::cout << "Mass Range: [" << config.massRange[0] << ", " << config.massRange[1] << "]"<< std::endl;
        std::cout << "Position Range: [" << config.positionRange[0] << ", " << config.positionRange[1] << "]"<< std::endl;
        std::cout << "Velocity Range: [" << config.velocityRange[0] << ", " << config.velocityRange[1] << "]"<< std::endl;
        std::cout << "Time Step: " << config.timeStep << "" << std::endl;
        std::cout << "Max Time: " << config.maxTime << "" << std::endl;
        // std::cout << "Number of Particles: " << config.numParticles << "" << std::endl;
        std::cout << "Number of Steps: " << config.nSteps << "" << std::endl;
        std::cout << "Use Barnes-Hut Algorithm: " << std::boolalpha << config.barnesHutAlgorithm << "" << std::endl;
        std::cout << "MAC: " << config.MAC << "" << std::endl;
        std::cout << "Tolerance for Unittest: " << config.unittestTolerance << "" << std::endl;

    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to parse config: ") + e.what());
    }

    return config;
}

// Read in File into a String
std::string readFile(const std::string &filePath) {

    std::ifstream file(filePath);

    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filePath);
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}


///Parsing functions for different Datatypes

// Parses an array from the JSON content
std::vector<double> parseArray(const std::string &key, const std::string &json) {

    std::size_t startPos = json.find(key);

    if (startPos == std::string::npos) {
        throw std::runtime_error("Key not found: " + key);
    }

    startPos = json.find('[', startPos);
    std::size_t endPos = json.find(']', startPos);

    if (startPos == std::string::npos || endPos == std::string::npos) {
        throw std::runtime_error("Malformed array for key: " + key);
    }

    std::string arrayContent = json.substr(startPos + 1, endPos - startPos - 1);

    std::vector<double> result;
    std::istringstream iss(arrayContent);

    double value;
    char comma;

    while (iss >> value) {
        result.push_back(value);
        iss >> comma;
    }

    return result;
}

// Parses a double value from the JSON content
double parseDouble(const std::string &key, const std::string &json) {

    std::size_t startPos = json.find(key);

    if (startPos == std::string::npos) {
        throw std::runtime_error("Key not found: " + key);
    }

    startPos = json.find(':', startPos);
    std::size_t endPos = json.find(',', startPos);

    if (endPos == std::string::npos) {
        endPos = json.find('}', startPos);
    }

    std::string valueStr = json.substr(startPos + 1, endPos - startPos - 1);
    return std::stod(valueStr);
}

// Parses a boolean value from the JSON content
bool parseBoolean(const std::string &key, const std::string &json) {
    std::size_t startPos = json.find(key);
    if (startPos == std::string::npos) {
        throw std::runtime_error("Key not found: " + key);
    }

    startPos = json.find(':', startPos);
    std::size_t endPos = json.find(',', startPos);
    if (endPos == std::string::npos) {
        endPos = json.find('}', startPos);
    }

    std::string valueStr = json.substr(startPos + 1, endPos - startPos - 1);
    valueStr.erase(remove_if(valueStr.begin(), valueStr.end(), ::isspace), valueStr.end());
    return valueStr == "true";
}

// Parse a scenario from scenarios.json and return particles
std::vector<Particle> parseScenario(const std::string& scenariosFilePath, const std::string& scenarioName) {
    std::vector<Particle> particles;

    try {
        std::string jsonContent = readFile(scenariosFilePath);

        // Find the scenario by name
        std::size_t scenarioPos = jsonContent.find("\"" + scenarioName + "\"");
        if (scenarioPos == std::string::npos) {
            throw std::runtime_error("Scenario not found: " + scenarioName);
        }

        // Find the particles array for this scenario
        std::size_t particlesPos = jsonContent.find("\"particles\"", scenarioPos);
        if (particlesPos == std::string::npos) {
            throw std::runtime_error("No particles found in scenario: " + scenarioName);
        }

        // Find the opening bracket of particles array
        std::size_t arrayStart = jsonContent.find('[', particlesPos);
        if (arrayStart == std::string::npos) {
            throw std::runtime_error("Malformed particles array in scenario: " + scenarioName);
        }

        // Find matching closing bracket (handle nested objects)
        int bracketCount = 1;
        std::size_t arrayEnd = arrayStart + 1;
        while (bracketCount > 0 && arrayEnd < jsonContent.size()) {
            if (jsonContent[arrayEnd] == '[') bracketCount++;
            else if (jsonContent[arrayEnd] == ']') bracketCount--;
            arrayEnd++;
        }

        std::string particlesArray = jsonContent.substr(arrayStart, arrayEnd - arrayStart);

        // Parse each particle object
        std::size_t particleStart = 0;
        while ((particleStart = particlesArray.find('{', particleStart)) != std::string::npos) {
            std::size_t particleEnd = particlesArray.find('}', particleStart);
            if (particleEnd == std::string::npos) break;

            std::string particleJson = particlesArray.substr(particleStart, particleEnd - particleStart + 1);

            // Parse position
            std::vector<double> posVec = parseArray("position", particleJson);
            std::array<double, 3> position = {posVec[0], posVec[1], posVec[2]};

            // Parse velocity
            std::vector<double> velVec = parseArray("velocity", particleJson);
            std::array<double, 3> velocity = {velVec[0], velVec[1], velVec[2]};

            // Parse mass
            double mass = parseDouble("mass", particleJson);

            particles.emplace_back(position, velocity, mass);

            particleStart = particleEnd + 1;
        }

        std::cout << "Loaded scenario '" << scenarioName << "' with " << particles.size() << " particles." << std::endl;

    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to parse scenario: ") + e.what());
    }

    return particles;
}

// Get list of available scenario names
std::vector<std::string> getAvailableScenarios(const std::string& scenariosFilePath) {
    std::vector<std::string> scenarios;

    try {
        std::string jsonContent = readFile(scenariosFilePath);

        // Find "scenarios" object
        std::size_t scenariosPos = jsonContent.find("\"scenarios\"");
        if (scenariosPos == std::string::npos) {
            return scenarios;
        }

        // Find the opening brace of scenarios object
        std::size_t objStart = jsonContent.find('{', scenariosPos);
        if (objStart == std::string::npos) {
            return scenarios;
        }

        // Look for scenario names (keys that are followed by an object with "description")
        std::size_t searchPos = objStart;
        while ((searchPos = jsonContent.find("\"description\"", searchPos)) != std::string::npos) {
            // Go backwards to find the scenario name
            std::size_t colonPos = jsonContent.rfind(':', searchPos);
            std::size_t bracePos = jsonContent.rfind('{', searchPos);

            if (bracePos != std::string::npos && bracePos > objStart) {
                // Find the key before this brace
                std::size_t keyEnd = jsonContent.rfind('"', bracePos - 1);
                std::size_t keyStart = jsonContent.rfind('"', keyEnd - 1);

                if (keyStart != std::string::npos && keyEnd != std::string::npos) {
                    std::string scenarioName = jsonContent.substr(keyStart + 1, keyEnd - keyStart - 1);
                    // Avoid duplicates
                    if (std::find(scenarios.begin(), scenarios.end(), scenarioName) == scenarios.end()) {
                        scenarios.push_back(scenarioName);
                    }
                }
            }
            searchPos++;
        }

    } catch (const std::exception& e) {
        std::cerr << "Warning: Could not read scenarios file: " << e.what() << std::endl;
    }

    return scenarios;
}

// Print usage information
void printUsage(const std::string& programName) {
    std::cout << "\nUsage: " << programName << " [OPTIONS]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --random, -r          Use random particles (default)\n";
    std::cout << "  --scenario, -s NAME   Use predefined scenario (e.g., solar_system, binary_star)\n";
    std::cout << "  --list, -l            List all available scenarios\n";
    std::cout << "  --tree-level N        Octree parallel construction level (1-4):\n";
    std::cout << "  --simd-batches N      Number of SIMD batches (8*<simd_batch>) for force calculation\n";
    std::cout << "                          1 = 8 subtrees, 2 = 64, 3 = 512, 4 = 4096\n";
    std::cout << "  --help, -h            Show this help message\n";
    std::cout << "\nAvailable scenarios:\n";
    std::cout << "  solar_system    - Simplified solar system with Sun and 4 planets\n";
    std::cout << "  binary_star     - Two stars orbiting each other with a planet\n";
    std::cout << "  galaxy_collision - Two galaxy cores approaching each other\n";
    std::cout << "  figure_eight    - Three-body figure-eight periodic orbit\n";
    std::cout << "  cluster         - Central mass with orbiting bodies\n";
    std::cout << "\nExamples:\n";
    std::cout << "  " << programName << " --random\n";
    std::cout << "  " << programName << " --scenario solar_system\n";
    std::cout << "  " << programName << " -s binary_star\n";
    std::cout << "  " << programName << " -n 1000000 --tree-level 3\n";
    std::cout << std::endl;
}
