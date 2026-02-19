// bruteForce.h
#ifndef BRUTE_FORCE_H
#define BRUTE_FORCE_H

#include <array>
#include <vector>
#include "ParticleData.h"


// SOA versions for better cache performance
void computeGravitationalForcesSOA(
    const ParticleData& particles,
    std::vector<double>& forceX,
    std::vector<double>& forceY,
    std::vector<double>& forceZ);

void updateParticlesSOA(
    ParticleData& particles,
    const std::vector<double>& forceX,
    const std::vector<double>& forceY,
    const std::vector<double>& forceZ,
    double timeStep);

#endif  // BRUTE_FORCE_H