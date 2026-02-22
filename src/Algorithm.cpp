#include "Algorithm.h"
#include "MeshTree.h"
#include "cubeNode.h"
#include "ParticleData.h"
#include <memory>
#include <cmath>
#include <algorithm>
#include <omp.h>

// Constructor sets interactionForce and MAC parameter
Algorithm::Algorithm(double macValue, InteractionForce& interactionForce) : c_MAC{macValue}, m_interactionForce{interactionForce} {}

// Helper: Collect all cubeNodes level by level for bottom-up processing
void Algorithm::collectNodesByLevel(Node* root, std::vector<std::vector<cubeNode*>>& levels)
{
    if (root == nullptr || root->isParticleNode()) {
        return;
    }

    std::vector<Node*> currentLevel;
    currentLevel.push_back(root);

    while (!currentLevel.empty()) {
        std::vector<cubeNode*> cubeNodesAtLevel;
        std::vector<Node*> nextLevel;

        for (Node* node : currentLevel) {
            if (!node->isParticleNode()) {
                cubeNode* cn = static_cast<cubeNode*>(node);
                cubeNodesAtLevel.push_back(cn);

                // Collect children for next level
#pragma unroll 8
                for (int i = 0; i < 8; ++i) {
                    Node* child = cn->getChild(i);
                    if (child != nullptr && !child->isParticleNode()) {
                        nextLevel.push_back(child);
                    }
                }
            }
        }

        if (!cubeNodesAtLevel.empty()) {
            levels.push_back(std::move(cubeNodesAtLevel));
        }
        currentLevel = std::move(nextLevel);
    }
}

// Set the properties of each CubeNode using bottom-up parallel accumulation
void Algorithm::precomputeCubeNodeProperties(Node* node)
{
    if (node == nullptr || node->isParticleNode()) {
        // Particle nodes don't need mass distribution calculation
        return;
    }

    // Collect all cubeNodes by level (level 0 = root, higher levels = deeper)
    std::vector<std::vector<cubeNode*>> levels;
    collectNodesByLevel(node, levels);

    if (levels.empty()) {
        return;
    }

    // Process bottom-up (leaf nodes first)
    for (int levelIdx = static_cast<int>(levels.size()) - 1; levelIdx >= 0; --levelIdx) {
        std::vector<cubeNode*>& nodesAtLevel = levels[levelIdx];

        // Process all nodes at this level in parallel
        #pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < nodesAtLevel.size(); ++i) {
            cubeNode* cn = nodesAtLevel[i];
            auto calcOutput = calculateCenterOfMassAndMass(cn);
            Cube currentCube = cn->getCube();
            currentCube.setCenterOfMass(calcOutput.first);
            currentCube.setMass(calcOutput.second);
            cn->setCube(currentCube);
        }
    }
}

// Calculate center of mass and total mass for a given cubeNode
// the children must already have their properties computed
std::pair<std::array<double, 3>, double> Algorithm::calculateCenterOfMassAndMass(cubeNode* currentCubeNode)
{
    std::array<double, 3> weightedPos = {0.0, 0.0, 0.0};
    double totalMass = 0.0;

    // Accumulate from all 8 children
#pragma unroll 8
    for (int i = 0; i < 8; ++i)
    {
        Node* childNode = currentCubeNode->getChild(i);
        if (childNode == nullptr) continue;

        if (childNode->isParticleNode())
        {
            particleNode* pn = static_cast<particleNode*>(childNode);
            if (pn->getParticle().has_value())
            {
                const Particle& particle = pn->getParticle().value();
                const double mass = particle.getMass();
                const auto& position = particle.getPosition();

                totalMass += mass;
                weightedPos[0] += position[0] * mass;
                weightedPos[1] += position[1] * mass;
                weightedPos[2] += position[2] * mass;
            }
        }
        else
        {
            // Child is a cubeNode - use its already computed mass and center of mass
            cubeNode* childCube = static_cast<cubeNode*>(childNode);
            const Cube& childCubeData = childCube->getCube();
            const double childMass = childCubeData.getMass();

            if (childMass > 0.0) {
                const auto& childCOM = childCubeData.getCenterOfMass();
                totalMass += childMass;
                weightedPos[0] += childCOM[0] * childMass;
                weightedPos[1] += childCOM[1] * childMass;
                weightedPos[2] += childCOM[2] * childMass;
            }
        }
    }

    // Calculate center of mass
    std::array<double, 3> centerOfMass = {0.0, 0.0, 0.0};
    if (totalMass > 0.0) {
        centerOfMass[0] = weightedPos[0] / totalMass;
        centerOfMass[1] = weightedPos[1] / totalMass;
        centerOfMass[2] = weightedPos[2] / totalMass;
    }

    return {centerOfMass, totalMass};
}

// Determine if a cluster (cube) is sufficiently far away from a particle
// to use the approximation based on the MAC criterion
bool Algorithm::isClusterSufficientlyFarAwaySOA(const Cube& currentCube, size_t particleIndex, const ParticleData& particles)
{
    constexpr double minDistanceSquared = 1e-10;

    const double regionWidth = currentCube.getNorm() * 2;
    const auto& centerOfMass = currentCube.getCenterOfMass();

    const double dx = centerOfMass[0] - particles.posX[particleIndex];
    const double dy = centerOfMass[1] - particles.posY[particleIndex];
    const double dz = centerOfMass[2] - particles.posZ[particleIndex];
    const double regionDistanceSquared = dx * dx + dy * dy + dz * dz;
    if (regionDistanceSquared > minDistanceSquared)
    {
        return (regionWidth * regionWidth / regionDistanceSquared) < c_MAC;
    }
    return false;
}

// run the Barnes-Hut algorithm using SOA data layout with SIMD optimizations
void Algorithm::runSOA_SIMD(Node* treeRoot, const ParticleData& particles,
                            std::vector<double>& forceX, std::vector<double>& forceY, std::vector<double>& forceZ)
{
    size_t n = particles.size();
    forceX.assign(n, 0.0);
    forceY.assign(n, 0.0);
    forceZ.assign(n, 0.0);

    cubeNode* rootNode = static_cast<cubeNode*>(treeRoot);
    precomputeCubeNodeProperties(rootNode);

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < n; ++i)
    {
        std::array<double, 3> force = calculateClusteredForcesSOA_SIMD(i, particles, rootNode);
        forceX[i] = force[0];
        forceY[i] = force[1];
        forceZ[i] = force[2];
    }
}

// Helper method that collects interactions into buffer and processes them with SIMD
void Algorithm::collectInteractionsSOA(size_t particleIndex, const ParticleData& particles,
                                       const std::array<double, 3>& particlePos, Node* node,
                                       InteractionBuffer& buffer, std::array<double, 3>& forceAccum)
{
    if (node->isParticleNode())
    {
        particleNode* currentParticleNode = static_cast<particleNode*>(node);
        const auto& currentParticle = currentParticleNode->getParticle();

        if (currentParticle.has_value())
        {
            const Particle& otherParticle = currentParticle.value();
            const auto& otherPos = otherParticle.getPosition();

            // Check if not the same particle (compare positions)
            if (otherPos[0] != particlePos[0] ||
                otherPos[1] != particlePos[1] ||
                otherPos[2] != particlePos[2])
            {
                // Add to buffer
                buffer.add(otherPos[0], otherPos[1], otherPos[2], otherParticle.getMass());

                // If buffer is full, process it with SIMD
                if (buffer.isFull())
                {
                    const double particleMass = particles.mass[particleIndex];
                    auto simdForce = m_interactionForce.calculateForceSIMD(particlePos, particleMass, buffer);
                    forceAccum[0] += simdForce[0];
                    forceAccum[1] += simdForce[1];
                    forceAccum[2] += simdForce[2];
                    buffer.clear();
                }
            }
        }
    }
    else
    {
        cubeNode* currentCubeNode = static_cast<cubeNode*>(node);
        Cube currentCube = currentCubeNode->getCube();

        if (isClusterSufficientlyFarAwaySOA(currentCube, particleIndex, particles))
        {
            const auto& com = currentCube.getCenterOfMass();
            buffer.add(com[0], com[1], com[2], currentCube.getMass());

            // If buffer is full, process it with SIMD
            if (buffer.isFull())
            {
                const double particleMass = particles.mass[particleIndex];
                auto simdForce = m_interactionForce.calculateForceSIMD(particlePos, particleMass, buffer);
                forceAccum[0] += simdForce[0];
                forceAccum[1] += simdForce[1];
                forceAccum[2] += simdForce[2];
                buffer.clear();
            }
        }
        else
        {
#pragma unroll 8
            for (int i = 0; i < 8; ++i)
            {
                Node* childNode = currentCubeNode->getChild(i);
                collectInteractionsSOA(particleIndex, particles, particlePos, childNode, buffer, forceAccum);
            }
        }
    }
}

// SIMD-optimized version that uses buffered force calculations
std::array<double, 3> Algorithm::calculateClusteredForcesSOA_SIMD(size_t particleIndex, const ParticleData& particles, Node* node)
{
    std::array<double, 3> forceAccum = {0.0, 0.0, 0.0};

    // Cache particle position for this index
    const std::array<double, 3> particlePos = {
        particles.posX[particleIndex],
        particles.posY[particleIndex],
        particles.posZ[particleIndex]
    };
    const double particleMass = particles.mass[particleIndex];

    // Thread-local buffer for collecting interactions
    InteractionBuffer buffer;

    // Traverse tree and collect interactions into buffer, processing when full
    collectInteractionsSOA(particleIndex, particles, particlePos, node, buffer, forceAccum);

    // Process any remaining interactions in the buffer (less than SIMD_WIDTH)
    if (buffer.count > 0)
    {
        // Use SIMD even for partial buffer - the SIMD function handles the count
        auto simdForce = m_interactionForce.calculateForceSIMD(particlePos, particleMass, buffer);
        forceAccum[0] += simdForce[0];
        forceAccum[1] += simdForce[1];
        forceAccum[2] += simdForce[2];
    }

    return forceAccum;
}

