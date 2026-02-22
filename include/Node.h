#ifndef BARNESHUT_NODE_H
#define BARNESHUT_NODE_H


#include <iostream>  // For std::cout (used in printInfo)
#include "memory"
#include "Cube.h"
class Node {

private:
    Cube m_cube;

protected:
    Node* m_parent;

public:
    //Constructor
    explicit Node(Node* parent, Cube cube);

    // Virtual destructor (needed for polymorphic deletion)
    virtual ~Node();

    [[nodiscard]] virtual bool isParticleNode() const { return false; }

    //All node types proper methods
    void insert(Node* parent);


    // Getter and setter for m_parent
    Node* getParent() const;
    void setParent(Node* p);

    //Getter and setter for Cube
    [[nodiscard]] const Cube &getCube() const;
    void setCube(const Cube &cube);
};

#endif //BARNESHUT_NODE_H
