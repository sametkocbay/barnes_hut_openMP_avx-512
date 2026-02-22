#include "../include/initialization.h"
#include <random>
#include "Cube.h"
#include "Particle.h"
#include "ParticleData.h"

// Generate random Particles using SOA layout
ParticleData generateRandomParticlesSOA(
    int numParticles,
    const std::array<double, 2>& massRange,
    const std::array<double, 2>& positionRange,
    const std::array<double, 2>& velocityRange
) {
    std::random_device rd;
    std::mt19937 gen(rd());

    // Create distributions for mass, position, and velocity
    std::uniform_real_distribution<> massDist(massRange[0], massRange[1]);
    std::uniform_real_distribution<> posDist(positionRange[0], positionRange[1]);
    std::uniform_real_distribution<> velDist(velocityRange[0], velocityRange[1]);

    ParticleData particles;
    particles.reserve(numParticles);

    for (int i = 0; i < numParticles; ++i) {
        particles.addParticle(
            posDist(gen), posDist(gen), posDist(gen),  // position x, y, z
            velDist(gen), velDist(gen), velDist(gen),  // velocity x, y, z
            massDist(gen)                               // mass
        );
    }

    return particles;
}

Cube calculateFirstCubeSOA(const ParticleData& particles) {
    double minX = std::numeric_limits<double>::infinity();
    double minY = std::numeric_limits<double>::infinity();
    double minZ = std::numeric_limits<double>::infinity();

    double maxX = -std::numeric_limits<double>::infinity();
    double maxY = -std::numeric_limits<double>::infinity();
    double maxZ = -std::numeric_limits<double>::infinity();

    size_t n = particles.size();
    for (size_t i = 0; i < n; ++i) {
        minX = std::min(minX, particles.posX[i]);
        minY = std::min(minY, particles.posY[i]);
        minZ = std::min(minZ, particles.posZ[i]);
        maxX = std::max(maxX, particles.posX[i]);
        maxY = std::max(maxY, particles.posY[i]);
        maxZ = std::max(maxZ, particles.posZ[i]);
    }

    std::array<double, 3> centerPoint{};
    centerPoint[0] = (maxX + minX) / 2.0;
    centerPoint[1] = (maxY + minY) / 2.0;
    centerPoint[2] = (maxZ + minZ) / 2.0;

    double lenX = maxX - minX;
    double lenY = maxY - minY;
    double lenZ = maxZ - minZ;

    double maxLen = std::max(std::abs(lenX), std::abs(lenY));
    maxLen = std::max(maxLen, std::abs(lenZ));
    double norm = maxLen * 0.505;

    return {centerPoint, norm};
}

// Convert AOS to SOA (needed for loading scenarios)
ParticleData convertToSOA(const std::vector<Particle>& particles) {
    ParticleData data;
    data.reserve(particles.size());

    for (const auto& p : particles) {
        const auto& pos = p.getPosition();
        const auto& vel = p.getVelocity();
        data.addParticle(pos[0], pos[1], pos[2], vel[0], vel[1], vel[2], p.getMass());
    }

    return data;
}


// Function to write particle positions to a CSV file
void writeToCSV(const std::vector<std::vector<std::array<double, 3>>>& particlePositions, const std::string& filename) {
    std::ofstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Could not open file for writing.");
    }

    size_t numParticles = particlePositions[0].size();


    // Write header row with particle position columns for each timestep
    for (size_t i = 0; i < numParticles; ++i) {
        file << "particle" << i << "X, particle" << i << "Y, particle" << i << "Z";
        if (i != numParticles - 1) {
            file << ", "; // Comma between particles
        }
    }
    file << "\n";

    size_t numTimesteps = particlePositions.size();

    // Write position data for each timestep
    for (size_t t = 0; t < numTimesteps; ++t) {
        for (size_t i = 0; i < numParticles; ++i) {
            const auto& position = particlePositions[t][i];
            file << position[0] << ", " << position[1] << ", " << position[2];
            if (i != numParticles - 1) {
                file << ", "; // Comma between particles
            }
        }
        file << "\n"; // Newline after each timestep
    }

    file.close(); // Close the file
}