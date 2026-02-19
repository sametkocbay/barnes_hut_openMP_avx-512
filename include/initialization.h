#ifndef INITIALIZATION_H
#define INITIALIZATION_H

#include <vector>
#include "Particle.h"
#include "ParticleData.h"
#include "Cube.h"
#include <string>
#include <iostream>
#include <fstream>

// Generate random Particles using SOA layout
ParticleData generateRandomParticlesSOA(
    int numParticles,
    const std::array<double, 2>& massRange,
    const std::array<double, 2>& positionRange,
    const std::array<double, 2>& velocityRange
);

Cube calculateFirstCubeSOA(const ParticleData& particles);

void writeToCSV(const std::vector<std::vector<std::array<double, 3>>>& particlePositions, const std::string& filename);

// Convert AOS to SOA (needed for loading scenarios)
ParticleData convertToSOA(const std::vector<Particle>& particles);

#endif // INITIALIZATION_H