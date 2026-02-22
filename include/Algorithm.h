#ifndef BARNESHUT_ALGORITHM_H
#define BARNESHUT_ALGORITHM_H

#include "Particle.h"
#include "ParticleData.h"
#include "particleNode.h"
#include "cubeNode.h"
#include "Cube.h"
#include "MeshTree.h"
#include "GravitationalForce.h"


class Algorithm {

private:

    //Constant Variable set by Config,
    // defining the distance between cluster and particle as fraction of clusterWidth and distance
    const double c_MAC;

    // Interaction force for the force calculation between the particles (e.g. gravitational force)
    InteractionForce& m_interactionForce;

    // Helper method for SIMD tree traversal - collects interactions into buffer
    void collectInteractionsSOA(size_t particleIndex, const ParticleData& particles,
                                const std::array<double, 3>& particlePos, Node* node,
                                InteractionBuffer& buffer, std::array<double, 3>& forceAccum);

public:

    // Constructor
    explicit Algorithm(double macValue, InteractionForce& interactionForce);

    // SIMD-optimized version using buffered force calculations
    void runSOA_SIMD(Node* treeRoot, const ParticleData& particles,
                     std::vector<double>& forceX, std::vector<double>& forceY, std::vector<double>& forceZ);

    // SIMD-optimized version that uses buffered force calculations
    std::array<double, 3> calculateClusteredForcesSOA_SIMD(size_t particleIndex, const ParticleData& particles, Node* node);

    std::pair<std::array<double, 3>, double> calculateCenterOfMassAndMass(cubeNode* currentCubeNode);
    void precomputeCubeNodeProperties(Node* node);
    bool isClusterSufficientlyFarAwaySOA(const Cube& currentCube, size_t particleIndex, const ParticleData& particles);
    void collectNodesByLevel(Node* root, std::vector<std::vector<cubeNode*>>& levels);
};


#endif //BARNESHUT_ALGORITHM_H