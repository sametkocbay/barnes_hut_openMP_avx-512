#ifndef BARNESHUT_CUBENODE_H
#define BARNESHUT_CUBENODE_H


#include <vector>
#include "Node.h"  // Include Node.h to inherit from Node
#include "Cube.h"
#include "Particle.h"

class cubeNode : public Node {


protected:
    std::array<Node*, 8> m_children;  // Array of child nodes

public:

    //Constructor
    cubeNode(Node *parent, const Cube &cube);

    //Destructor
    ~cubeNode() override;

    //Class specific Functions [Methods only for internal nodes]
    int addChild(Node* child);
    void createNewChildLevel(const std::array<Cube, 8>& cubes);

    //Getter and Setter
    Node* getChild(int index) const ;
    void setChild(int index, Node* child);
};


#endif //BARNESHUT_CUBENODE_H
