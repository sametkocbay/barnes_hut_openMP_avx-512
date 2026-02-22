#ifndef INTERACTION_FORCE_H
#define INTERACTION_FORCE_H

#include <vector>
#include <array>
#include "Particle.h"

// SIMD buffer size for AVX-512 (8 doubles)
constexpr size_t SIMD_LANES = 8;
constexpr size_t MAX_SIMD_BATCHES = 16;  // Maximum batches at compile-time
constexpr size_t MAX_SIMD_WIDTH = SIMD_LANES * MAX_SIMD_BATCHES;

// Runtime configurable SIMD batches
inline size_t SIMD_BATCHES = 1;
inline size_t SIMD_WIDTH = SIMD_LANES * SIMD_BATCHES;

inline void setSIMDBatches(size_t batches) {
    SIMD_BATCHES = (batches > 0 && batches <= MAX_SIMD_BATCHES) ? batches : 1;
    SIMD_WIDTH = SIMD_LANES * SIMD_BATCHES;
}

// Structure to hold interaction data for SIMD processing
struct alignas(64) InteractionBuffer {
    double otherPosX[MAX_SIMD_WIDTH] = {0.0};
    double otherPosY[MAX_SIMD_WIDTH] = {0.0};
    double otherPosZ[MAX_SIMD_WIDTH] = {0.0};
    double otherMass[MAX_SIMD_WIDTH] = {0.0};
    size_t count = 0;

    void clear() { count = 0; }
    [[nodiscard]] bool isFull() const { return count >= SIMD_WIDTH; }

    void add(double px, double py, double pz, double m) {
        otherPosX[count] = px;
        otherPosY[count] = py;
        otherPosZ[count] = pz;
        otherMass[count] = m;
        ++count;
    }
};

class InteractionForce {

public:
    virtual ~InteractionForce() = default;

    virtual std::array<double, 3> calculateForceSOA(
        const std::array<double, 3>& position, double mass,
        const std::array<double, 3>& otherPosition, double otherMass) = 0;

    // SIMD version: calculates forces for up to 8 interactions at once
    virtual std::array<double, 3> calculateForceSIMD(
        const std::array<double, 3>& position, double mass,
        const InteractionBuffer& buffer) = 0;
};

#endif //INTERACTION_FORCE_H
