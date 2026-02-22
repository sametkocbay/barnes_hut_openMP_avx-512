#ifndef GRAVITATIONAL_FORCE_H
#define GRAVITATIONAL_FORCE_H

#include "InteractionForce.h"

class GravitationalForce : public InteractionForce {

public:
    std::array<double, 3> calculateForceSOA(
        const std::array<double, 3>& position, double mass,
        const std::array<double, 3>& otherPosition, double otherMass) override;

    // AVX-512 SIMD version for processing 8 interactions at once
    std::array<double, 3> calculateForceSIMD(
        const std::array<double, 3>& position, double mass,
        const InteractionBuffer& buffer) override;
};

#endif //GRAVITATIONAL_FORCE_H
