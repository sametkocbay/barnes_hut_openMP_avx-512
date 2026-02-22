#include "GravitationalForce.h"
#include <cmath>
#ifdef __AVX512F__
#include <immintrin.h>
#endif

std::array<double, 3> GravitationalForce::calculateForceSOA(
    const std::array<double, 3>& position, double mass,
    const std::array<double, 3>& otherPosition, double otherMass)
{
    // Gravitational constant in m^3 kg^-1 s^-2
    constexpr double G = 6.67430e-11;

    std::array<double, 3> forceVector = {0.0, 0.0, 0.0};

    // Calc distance between particles
    double dx = otherPosition[0] - position[0];
    double dy = otherPosition[1] - position[1];
    double dz = otherPosition[2] - position[2];
    double distanceSquared = dx * dx + dy * dy + dz * dz;
    double distance = std::sqrt(distanceSquared);

    // If super small skip so it doesnt divide by zero
    if (distance < 0.5) return forceVector;

    // Gravitational force
    double forceMagnitude = (G * mass * otherMass) / distanceSquared;
    double invDistance = 1.0 / distance;

    // Apply force magnitude in the direction
    forceVector[0] = forceMagnitude * dx * invDistance;
    forceVector[1] = forceMagnitude * dy * invDistance;
    forceVector[2] = forceMagnitude * dz * invDistance;

    return forceVector;
}
std::array<double, 3> GravitationalForce::calculateForceSIMD(
    const std::array<double, 3>& position, double mass,
    const InteractionBuffer& buffer)
{
    constexpr double G = 6.67430e-11;
    constexpr double minDistance = 0.5;

    std::array<double, 3> totalForce = {0.0, 0.0, 0.0};

    if (buffer.count == 0) return totalForce;

#ifdef __AVX512F__
    __m512d posX = _mm512_set1_pd(position[0]);
    __m512d posY = _mm512_set1_pd(position[1]);
    __m512d posZ = _mm512_set1_pd(position[2]);
    __m512d particleMass = _mm512_set1_pd(mass);
    __m512d gVec = _mm512_set1_pd(G);
    __m512d minDistVec = _mm512_set1_pd(minDistance);

    // Schleife über alle 8er-Batches
    size_t numBatches = (buffer.count + SIMD_LANES - 1) / SIMD_LANES;

    for (size_t batch = 0; batch < numBatches; ++batch) {
        size_t offset = batch * SIMD_LANES;
        size_t remaining = buffer.count - offset;

        __m512d otherX = _mm512_loadu_pd(&buffer.otherPosX[offset]);
        __m512d otherY = _mm512_loadu_pd(&buffer.otherPosY[offset]);
        __m512d otherZ = _mm512_loadu_pd(&buffer.otherPosZ[offset]);
        __m512d otherMass = _mm512_loadu_pd(&buffer.otherMass[offset]);

        __m512d dx = _mm512_sub_pd(otherX, posX);
        __m512d dy = _mm512_sub_pd(otherY, posY);
        __m512d dz = _mm512_sub_pd(otherZ, posZ);

        __m512d distSq = _mm512_add_pd(
            _mm512_add_pd(_mm512_mul_pd(dx, dx), _mm512_mul_pd(dy, dy)),
            _mm512_mul_pd(dz, dz)
        );

        __m512d dist = _mm512_sqrt_pd(distSq);

        __mmask8 validMask = _mm512_cmp_pd_mask(dist, minDistVec, _CMP_GE_OQ);
        __mmask8 countMask = (remaining >= SIMD_LANES) ? 0xFF : ((1 << remaining) - 1);
        validMask = validMask & countMask;

        __m512d forceMag = _mm512_div_pd(
            _mm512_mul_pd(_mm512_mul_pd(gVec, particleMass), otherMass),
            distSq
        );

        __m512d invDist = _mm512_div_pd(_mm512_set1_pd(1.0), dist);

        __m512d forceX = _mm512_maskz_mov_pd(validMask, _mm512_mul_pd(_mm512_mul_pd(forceMag, dx), invDist));
        __m512d forceY = _mm512_maskz_mov_pd(validMask, _mm512_mul_pd(_mm512_mul_pd(forceMag, dy), invDist));
        __m512d forceZ = _mm512_maskz_mov_pd(validMask, _mm512_mul_pd(_mm512_mul_pd(forceMag, dz), invDist));

        totalForce[0] += _mm512_reduce_add_pd(forceX);
        totalForce[1] += _mm512_reduce_add_pd(forceY);
        totalForce[2] += _mm512_reduce_add_pd(forceZ);
    }
#else
    // Fallback bleibt gleich
    for (size_t i = 0; i < buffer.count; ++i) {
        std::array<double, 3> otherPos = {buffer.otherPosX[i], buffer.otherPosY[i], buffer.otherPosZ[i]};
        auto force = calculateForceSOA(position, mass, otherPos, buffer.otherMass[i]);
        totalForce[0] += force[0];
        totalForce[1] += force[1];
        totalForce[2] += force[2];
    }
#endif

    return totalForce;
}


