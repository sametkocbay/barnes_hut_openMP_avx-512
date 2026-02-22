#ifndef BARNESHUT_MESHTREE_H
#define BARNESHUT_MESHTREE_H

#include "Node.h"
#include "Cube.h"
#include "cubeNode.h"
#include "Particle.h"
#include "ParticleData.h"
#include <vector>
#include "particleNode.h"
#include "SpaceDividing.h"

class MeshTree : public SpaceDividing {

private:
    Node* m_root;
    int m_parallelLevel;  // 1=8, 2=64, 3=512, 4=4096 subtrees

public:

    //Create Tree
    explicit MeshTree(Cube rootCube, int parallelLevel = 2);
    Node* divideSpaceSOA(const ParticleData &particles) override;
    Node* createMeshSOA(const ParticleData &particles);
    Node* createMeshSOAParallel(const ParticleData &particles);

    //Destructor
    ~MeshTree();
    void destroyTreeParallel();

    //Tree methods (SOA)
    std::pair<Node*, int> searchTreeSOA(size_t particleIndex, const ParticleData &particles, Node *startNode, int index = 0);
    void treeCreationSOA(size_t particleIndex, const ParticleData &particles, Node *node);
    void treeCreationSOASubtree(size_t particleIndex, const ParticleData &particles, cubeNode* parentCubeNode, int childIdx);
    bool particleFitsInCubeSOA(size_t particleIndex, const ParticleData& particles, const Cube& cube);

    // Helper for parallel tree construction
    int determineSubCubeIndex(size_t particleIndex, const ParticleData& particles, const Cube& parentCube);

    std::array<Cube, 8> calculateCubeProperties(const Cube& cube);
    cubeNode* replaceParticleNodeCubeNode(const std::array<Cube, 8> &newCubes, const int childNr, particleNode* currentParticleNode);
    cubeNode* replaceParticleNodeCubeNodeSubtree(const std::array<Cube, 8> &newCubes, const int childNr,
                                                  particleNode* currentParticleNode, cubeNode* rootCubeNode, int subtreeIdx);

    //Getter and Setter
    Node* getRoot();
};

#endif //BARNESHUT_MESHTREE_H
