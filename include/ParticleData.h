#ifndef BARNESHUT_PARTICLEDATA_H
#define BARNESHUT_PARTICLEDATA_H

#include <vector>
#include <array>
#include <cstddef>
#include <stdexcept>

/**
 * Structure of Arrays (SOA) for particle data.
 * This layout is more cache-friendly for vectorized computations
 * as it stores each property in contiguous memory.
 */
struct ParticleData {
    // Position components (x, y, z)
    std::vector<double> posX;
    std::vector<double> posY;
    std::vector<double> posZ;

    // Velocity components (x, y, z)
    std::vector<double> velX;
    std::vector<double> velY;
    std::vector<double> velZ;

    // Mass
    std::vector<double> mass;

    // Default constructor
    ParticleData() = default;

    // Constructor with pre-allocated size
    explicit ParticleData(size_t numParticles) {
        resize(numParticles);
    }

    // Get number of particles
    [[nodiscard]] size_t size() const {
        return mass.size();
    }

    // Check if empty
    [[nodiscard]] bool empty() const {
        return mass.empty();
    }

    // Resize all arrays
    void resize(size_t numParticles) {
        posX.resize(numParticles);
        posY.resize(numParticles);
        posZ.resize(numParticles);
        velX.resize(numParticles);
        velY.resize(numParticles);
        velZ.resize(numParticles);
        mass.resize(numParticles);
    }

    // Reserve capacity in all arrays
    void reserve(size_t capacity) {
        posX.reserve(capacity);
        posY.reserve(capacity);
        posZ.reserve(capacity);
        velX.reserve(capacity);
        velY.reserve(capacity);
        velZ.reserve(capacity);
        mass.reserve(capacity);
    }

    // Clear all arrays
    void clear() {
        posX.clear();
        posY.clear();
        posZ.clear();
        velX.clear();
        velY.clear();
        velZ.clear();
        mass.clear();
    }

    // Add a particle
    void addParticle(double px, double py, double pz,
                     double vx, double vy, double vz,
                     double m) {
        posX.push_back(px);
        posY.push_back(py);
        posZ.push_back(pz);
        velX.push_back(vx);
        velY.push_back(vy);
        velZ.push_back(vz);
        mass.push_back(m);
    }

    // Set particle at index
    void setParticle(size_t index, double px, double py, double pz,
                     double vx, double vy, double vz,
                     double m) {
        posX[index] = px;
        posY[index] = py;
        posZ[index] = pz;
        velX[index] = vx;
        velY[index] = vy;
        velZ[index] = vz;
        mass[index] = m;
    }

    // Get position as array
    [[nodiscard]] std::array<double, 3> getPosition(size_t index) const {
        return {posX[index], posY[index], posZ[index]};
    }

    // Get velocity as array
    [[nodiscard]] std::array<double, 3> getVelocity(size_t index) const {
        return {velX[index], velY[index], velZ[index]};
    }

    // Set position
    void setPosition(size_t index, double px, double py, double pz) {
        posX[index] = px;
        posY[index] = py;
        posZ[index] = pz;
    }

    // Set velocity
    void setVelocity(size_t index, double vx, double vy, double vz) {
        velX[index] = vx;
        velY[index] = vy;
        velZ[index] = vz;
    }
};

#endif //BARNESHUT_PARTICLEDATA_H
