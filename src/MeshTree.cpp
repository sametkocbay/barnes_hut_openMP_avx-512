#include "MeshTree.h"
#include "Node.h"
#include "particleNode.h"
#include "Particle.h"
#include "ParticleData.h"
#include <optional>
#include "cubeNode.h"
#include "Cube.h"
#include <array>
#include <omp.h>
#include <functional>

MeshTree::MeshTree(Cube rootCube, int parallelLevel)  {
    m_root = new particleNode(nullptr, rootCube);
    m_parallelLevel = parallelLevel;
}

Node* MeshTree::divideSpaceSOA(const ParticleData &particles) {
    return createMeshSOAParallel(particles);
}

Node* MeshTree::createMeshSOA(const ParticleData &particles) {
    size_t n = particles.size();
    for (size_t i = 0; i < n; ++i) {
        treeCreationSOA(i, particles, m_root);
    }
    return m_root;
}

// Parallelized creation of the MeshTree using OpenMP tasks
// 1. Divide the root cube into 8^L subcubes (L = m_parallelLevel)
// 2. Sort particles into buckets based on which subcube they belong to
// 3. Build each subtree in parallel using OpenMP tasks, inserting particles from the corresponding bucket
// The larger the m_parallelLevel, the more it adapts to load imabalances due to uneven
// distribution of particles but also more overhead

Node* MeshTree::createMeshSOAParallel(const ParticleData &particles) {
    size_t n = particles.size();

    // Calculate number of subtrees based on parallel level
    int numSubtrees = 1;
    for (int i = 0; i < m_parallelLevel; ++i) numSubtrees *= 8;

    // For small particle counts, use sequential version
    if (n < static_cast<size_t>(numSubtrees)) {
        return createMeshSOA(particles);
    }

    Cube rootCube = m_root->getCube();

    // Build 8^L Subtrees in parallel using OpenMP
    std::vector<Cube> leafCubes(numSubtrees);

    // Recursively calculate all leaf cubes
    std::function<void(const Cube&, int, int)> calculateLeafCubes = [&](const Cube& cube, int level, int baseIdx) {
        if (level == m_parallelLevel) {
            leafCubes[baseIdx] = cube;
            return;
        }
        std::array<Cube, 8> subCubes = calculateCubeProperties(cube);
        int childMultiplier = 1;
        for (int i = level + 1; i < m_parallelLevel; ++i) childMultiplier *= 8;
        for (int i = 0; i < 8; ++i) {
            calculateLeafCubes(subCubes[i], level + 1, baseIdx + i * childMultiplier);
        }
    };
    calculateLeafCubes(rootCube, 0, 0);

    // Sort particles into buckets based on which leaf cube they belong to
    std::vector<std::vector<size_t>> particleIndices(numSubtrees);

    #pragma omp parallel
    {
        std::vector<std::vector<size_t>> particleIndices_local(numSubtrees);

        #pragma omp for nowait
        for (size_t i = 0; i < n; ++i) {
            // Traverse down the tree to find the correct leaf cube
            int flatIdx = 0;
            Cube currentCube = rootCube;
            bool valid = true;

            for (int level = 0; level < m_parallelLevel && valid; ++level) {
                int subIdx = determineSubCubeIndex(i, particles, currentCube);
                if (subIdx < 0 || subIdx >= 8) {
                    valid = false;
                    break;
                }

                int multiplier = 1;
                for (int l = level + 1; l < m_parallelLevel; ++l) multiplier *= 8;
                flatIdx += subIdx * multiplier;

                // Calculate the subcube for next iteration
                std::array<Cube, 8> subCubes = calculateCubeProperties(currentCube);
                currentCube = subCubes[subIdx];
            }

            if (valid) {
                particleIndices_local[flatIdx].push_back(i);
            }
        }

        #pragma omp critical
        {
            for (int i = 0; i < numSubtrees; ++i) {
                particleIndices[i].insert(
                    particleIndices[i].end(),
                    particleIndices_local[i].begin(),
                    particleIndices_local[i].end()
                );
            }
        }
    }

    // Build tree structure down to leaf level
    // Store leaf parent nodes and their child indices for task processing
    std::vector<cubeNode*> leafParents(numSubtrees);
    std::vector<int> leafChildIndices(numSubtrees);

    // Recursively build tree structure
    std::function<Node*(Node*, const Cube&, int, int)> buildTreeStructure = [&](Node* parent, const Cube& cube, int level, int baseIdx) -> Node* {
        if (level == m_parallelLevel) {
            // Leaf level - create particleNode
            particleNode* leaf = new particleNode(parent, cube);
            leafParents[baseIdx] = static_cast<cubeNode*>(parent);
            leafChildIndices[baseIdx] = baseIdx % 8;  // Child index within parent
            return leaf;
        }

        // Internal level -> create cubeNode
        cubeNode* node = new cubeNode(parent, cube);
        std::array<Cube, 8> subCubes = calculateCubeProperties(cube);

        int childMultiplier = 1;
        for (int i = level + 1; i < m_parallelLevel; ++i) childMultiplier *= 8;

        for (int i = 0; i < 8; ++i) {
            Node* child = buildTreeStructure(node, subCubes[i], level + 1, baseIdx + i * childMultiplier);
            node->setChild(i, child);
        }

        return node;
    };

    m_root = buildTreeStructure(nullptr, rootCube, 0, 0);

    // Build subtrees in parallel using OpenMP tasks
    #pragma omp parallel
    {
        #pragma omp single
        {
            for (int subtreeIdx = 0; subtreeIdx < numSubtrees; ++subtreeIdx) {
                if (!particleIndices[subtreeIdx].empty()) {
                    #pragma omp task firstprivate(subtreeIdx)
                    {
                        cubeNode* parentNode = leafParents[subtreeIdx];
                        int childIdx = leafChildIndices[subtreeIdx];
                        for (size_t idx : particleIndices[subtreeIdx]) {
                            treeCreationSOASubtree(idx, particles, parentNode, childIdx);
                        }
                    }
                }
            }
        }
    }

    return m_root;
}

int MeshTree::determineSubCubeIndex(size_t particleIndex, const ParticleData& particles, const Cube& parentCube) {
    const auto& center = parentCube.getCenterPoint();

    double px = particles.posX[particleIndex];
    double py = particles.posY[particleIndex];
    double pz = particles.posZ[particleIndex];

    int index = 0;
    if (px >= center[0]) index |= 4;
    if (py >= center[1]) index |= 2;
    if (pz >= center[2]) index |= 1;

    return index;
}

// Insert particle into subtree rooted at parentCubeNode at child index childIdx if Node is occupied convert to cubeNode
// and split further recursively via the classic recurisve method
void MeshTree::treeCreationSOASubtree(size_t particleIndex, const ParticleData &particles, cubeNode* parentCubeNode, int childIdx) {
    Node* subtreeRoot = parentCubeNode->getChild(childIdx);

    std::pair<Node*, int> searchTreeOutput = searchTreeSOA(particleIndex, particles, subtreeRoot);

    particleNode* currentNode = static_cast<particleNode*>(searchTreeOutput.first);
    int localChildNr = searchTreeOutput.second;

    int childNr = (currentNode == subtreeRoot) ? childIdx : localChildNr;

    if(currentNode->getParticle() != std::nullopt) {
        Particle currentParticle = currentNode->getParticle().value();

        std::array<Cube,8> newCubes = calculateCubeProperties(currentNode->getCube());

        cubeNode *newCubeNode = replaceParticleNodeCubeNodeSubtree(newCubes, childNr, currentNode, parentCubeNode, childIdx);

        std::pair<Node*, int> newSearchOutput = searchTreeSOA(particleIndex, particles, newCubeNode);
        particleNode* currentParticleNode = static_cast<particleNode*>(newSearchOutput.first);

        Particle newParticle(
            {particles.posX[particleIndex], particles.posY[particleIndex], particles.posZ[particleIndex]},
            {particles.velX[particleIndex], particles.velY[particleIndex], particles.velZ[particleIndex]},
            particles.mass[particleIndex]
        );
        currentParticleNode->setParticle(newParticle);

        ParticleData tempParticles;
        tempParticles.addParticle(
            currentParticle.getPosition()[0], currentParticle.getPosition()[1], currentParticle.getPosition()[2],
            currentParticle.getVelocity()[0], currentParticle.getVelocity()[1], currentParticle.getVelocity()[2],
            currentParticle.getMass()
        );
        treeCreationSOA(0, tempParticles, newCubeNode);
    }
    else {
        Particle newParticle(
            {particles.posX[particleIndex], particles.posY[particleIndex], particles.posZ[particleIndex]},
            {particles.velX[particleIndex], particles.velY[particleIndex], particles.velZ[particleIndex]},
            particles.mass[particleIndex]
        );
        currentNode->setParticle(newParticle);
    }
}

cubeNode* MeshTree::replaceParticleNodeCubeNodeSubtree(const std::array<Cube, 8> &newCubes, const int childNr,
                                                        particleNode* currentParticleNode, cubeNode* rootCubeNode, int subtreeIdx) {
    cubeNode* parentNode = static_cast<cubeNode*>(currentParticleNode->getParent());

    cubeNode* newCubeNode = new cubeNode(parentNode, currentParticleNode->getCube());
    currentParticleNode->setCube(newCubes[0]);
    currentParticleNode->setParticle(std::nullopt);
    currentParticleNode->setParent(newCubeNode);
    newCubeNode->setChild(0, currentParticleNode);

    for (int i = 1; i < 8; ++i) {
        particleNode* newChildNode = new particleNode(newCubeNode, newCubes[i]);
        newCubeNode->setChild(i, newChildNode);
    }

    if(parentNode != nullptr) {
        parentNode->setChild(childNr, newCubeNode);
    }
    else {
        rootCubeNode->setChild(subtreeIdx, newCubeNode);
        newCubeNode->setParent(rootCubeNode);
    }

    return newCubeNode;
}

void MeshTree::treeCreationSOA(size_t particleIndex, const ParticleData &particles, Node *node) {
    std::pair<Node*, int> searchTreeOutput = searchTreeSOA(particleIndex, particles, node);

    particleNode* currentNode = static_cast<particleNode*>(searchTreeOutput.first);
    int childNr = searchTreeOutput.second;

    if(currentNode->getParticle() != std::nullopt) {
        Particle currentParticle = currentNode->getParticle().value();

        std::array<Cube,8> newCubes = calculateCubeProperties(currentNode->getCube());

        cubeNode *newCubeNode = replaceParticleNodeCubeNode(newCubes, childNr, currentNode);

        // Setup first Particle from SOA
        std::pair<Node*, int> newSearchOutput = searchTreeSOA(particleIndex, particles, newCubeNode);
        particleNode* currentParticleNode = static_cast<particleNode*>(newSearchOutput.first);

        // Create Particle from SOA data and store it
        Particle newParticle(
            {particles.posX[particleIndex], particles.posY[particleIndex], particles.posZ[particleIndex]},
            {particles.velX[particleIndex], particles.velY[particleIndex], particles.velZ[particleIndex]},
            particles.mass[particleIndex]
        );
        currentParticleNode->setParticle(newParticle);

        // Recursively handle the existing particle using a temporary SOA for the single particle
        // Create temp particle and insert it into the tree
        ParticleData tempParticles;
        tempParticles.addParticle(
            currentParticle.getPosition()[0], currentParticle.getPosition()[1], currentParticle.getPosition()[2],
            currentParticle.getVelocity()[0], currentParticle.getVelocity()[1], currentParticle.getVelocity()[2],
            currentParticle.getMass()
        );
        treeCreationSOA(0, tempParticles, newCubeNode);
    }
    else {
        // Create Particle from SOA data and store it
        Particle newParticle(
            {particles.posX[particleIndex], particles.posY[particleIndex], particles.posZ[particleIndex]},
            {particles.velX[particleIndex], particles.velY[particleIndex], particles.velZ[particleIndex]},
            particles.mass[particleIndex]
        );
        currentNode->setParticle(newParticle);
    }
}

std::pair<Node*, int> MeshTree::searchTreeSOA(size_t particleIndex, const ParticleData &particles, Node *startNode, int index) {
    // If startNode is a particleNode type, return it
    if (startNode->isParticleNode()) {
        return {startNode, index};
    }
    // Cast StartNode to cubeNode
    cubeNode* currentCubeNode = static_cast<cubeNode*>(startNode);

    // Loop through the 8 childs of the cubeNode
    for(int i = 0; i < 8; i++) {
        Node* child = currentCubeNode->getChild(i);
        if(child != nullptr && particleFitsInCubeSOA(particleIndex, particles, child->getCube())) {
            return searchTreeSOA(particleIndex, particles, child, i);
        }
    }
    throw std::runtime_error("Could not find a matching cube for the particle.");
}

bool MeshTree::particleFitsInCubeSOA(size_t particleIndex, const ParticleData& particles, const Cube& cube) {
    // Get the cube's center and norm (half-length of the cube)
    const auto& cubeCenter = cube.getCenterPoint();
    double norm = cube.getNorm();

    // Check if the particle is within the bounds of the cube along all axes (x, y, z)
    return (particles.posX[particleIndex] >= (cubeCenter[0] - norm) && particles.posX[particleIndex] <= (cubeCenter[0] + norm)) &&
           (particles.posY[particleIndex] >= (cubeCenter[1] - norm) && particles.posY[particleIndex] <= (cubeCenter[1] + norm)) &&
           (particles.posZ[particleIndex] >= (cubeCenter[2] - norm) && particles.posZ[particleIndex] <= (cubeCenter[2] + norm));
}

cubeNode* MeshTree::replaceParticleNodeCubeNode(const std::array<Cube, 8> &newCubes, const int childNr, particleNode* currentParticleNode) {
    // Use the parent of the currentParticleNode to reattach the new cube node
    cubeNode* parentNode = static_cast<cubeNode*>(currentParticleNode->getParent());

    // create new cube node with children
    cubeNode* newCubeNode = new cubeNode(parentNode, currentParticleNode->getCube());
    currentParticleNode->setCube(newCubes[0]);
    currentParticleNode->setParticle(std::nullopt);
    currentParticleNode->setParent(newCubeNode);
    newCubeNode->setChild(0, currentParticleNode);
    for (int i = 1; i < 8; ++i) {
        particleNode* newChildNode = new particleNode(newCubeNode, newCubes[i]);
        newCubeNode->setChild(i, newChildNode);
    }

    // set newCubeNode as child for parent node, except for the first iteration where parent is nullptr
    if(parentNode != nullptr) {
        parentNode->setChild(childNr, newCubeNode);
    }
    else {
        m_root = newCubeNode;
    }

    // Return the newly created cube node
    return newCubeNode;
}

std::array<Cube, 8> MeshTree::calculateCubeProperties(const Cube& cube) {
    std::array<Cube, 8> subCubes;
    double norm = cube.getNorm();
    double newNorm = norm * 0.5;

    //Array for saving offset directions of every new subcube
    static const std::array<std::array<int, 3>, 8> offsets = {{
        {-1, -1, -1}, {-1, -1, 1}, {-1, 1, -1}, {-1, 1, 1},
        {1, -1, -1}, {1, -1, 1}, {1, 1, -1}, {1, 1, 1}
    }};

    const auto& parentCenter = cube.getCenterPoint();

    for (int i = 0; i < 8; ++i) {
        subCubes[i] = {
            { parentCenter[0] + offsets[i][0] * newNorm,
              parentCenter[1] + offsets[i][1] * newNorm,
              parentCenter[2] + offsets[i][2] * newNorm },
            newNorm
        };
    }

    return subCubes;
}

Node* MeshTree::getRoot() {
    return m_root;
}
void MeshTree::destroyTreeParallel() {
    if (m_root == nullptr) {
        return;
    }

    // If root is a particle node, just delete it directly
    if (m_root->isParticleNode()) {
        delete m_root;
        m_root = nullptr;
        return;
    }

    // Root is a cubeNode - parallelize deletion of the 8 subtrees
    cubeNode* rootCube = static_cast<cubeNode*>(m_root);

    // Parallel deletion of top-level subtrees (each thread handles one subtree)
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < 8; ++i) {
        Node* child = rootCube->getChild(i);
        if (child != nullptr) {
            delete child;  // Recursive deletion within each thread's subtree
        }
    }

    // Set children to nullptr so cubeNode destructor doesn't double-delete
    for (int i = 0; i < 8; ++i) {
        rootCube->setChild(i, nullptr);
    }

    // Now delete the root node itself
    delete rootCube;
    m_root = nullptr;
}

MeshTree::~MeshTree() {
    destroyTreeParallel();
}
