// bruteForce.cpp
#include "bruteForce.h"
#include "ParticleData.h"
#include <cmath>
#include <vector>

// Gravitational constant in m^3 kg^-1 s^-2
const double G = 6.67430e-11;

// SOA version: Compute gravitational forces for all particles
void computeGravitationalForcesSOA(
    const ParticleData& particles,
    std::vector<double>& forceX,
    std::vector<double>& forceY,
    std::vector<double>& forceZ)
{
    size_t n = particles.size();

    // Initialize force vectors
    forceX.assign(n, 0.0);
    forceY.assign(n, 0.0);
    forceZ.assign(n, 0.0);

    // Iterate over each pair of particles
    for (size_t i = 0; i < n; ++i) {
        const double posX_i = particles.posX[i];
        const double posY_i = particles.posY[i];
        const double posZ_i = particles.posZ[i];
        const double mass_i = particles.mass[i];

        for (size_t j = 0; j < n; ++j) {
            if (i == j) continue; // Skip itself

            // Calc distance between particles
            double dx = particles.posX[j] - posX_i;
            double dy = particles.posY[j] - posY_i;
            double dz = particles.posZ[j] - posZ_i;
            double distanceSquared = dx * dx + dy * dy + dz * dz;
            double distance = std::sqrt(distanceSquared);

            // If super small skip so it doesnt divide by zero
            if (distance < 0.5) continue;

            double invDistance = 1.0 / distance;

            // Gravitational force
            double forceMagnitude = (G * mass_i * particles.mass[j]) / distanceSquared;

            // Apply force magnitude in the direction for particle i
            forceX[i] += forceMagnitude * dx * invDistance;
            forceY[i] += forceMagnitude * dy * invDistance;
            forceZ[i] += forceMagnitude * dz * invDistance;
        }
    }
}

// SOA version: Update particle positions and velocities
void updateParticlesSOA(
    ParticleData& particles,
    const std::vector<double>& forceX,
    const std::vector<double>& forceY,
    const std::vector<double>& forceZ,
    double timeStep)
{
    size_t n = particles.size();

    for (size_t i = 0; i < n; ++i) {
        double mass = particles.mass[i];
        if (mass <= 0) continue;

        // Calculate acceleration
        double invMass = 1.0 / mass;
        double ax = forceX[i] * invMass;
        double ay = forceY[i] * invMass;
        double az = forceZ[i] * invMass;

        // Update velocity
        particles.velX[i] += ax * timeStep;
        particles.velY[i] += ay * timeStep;
        particles.velZ[i] += az * timeStep;

        // Update position
        particles.posX[i] += particles.velX[i] * timeStep;
        particles.posY[i] += particles.velY[i] * timeStep;
        particles.posZ[i] += particles.velZ[i] * timeStep;
    }
}
