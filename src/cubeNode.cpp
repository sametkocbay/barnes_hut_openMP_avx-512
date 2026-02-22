#include "cubeNode.h"
#include "Particle.h"
#include "particleNode.h"
#include "Node.h"


cubeNode::cubeNode(Node *parent, const Cube &cube) : Node(parent,cube), m_children{nullptr}
    {
        // Initialize all child nodes to nullptr
        for (int i = 0; i < 8; i++) {
            m_children[i] = nullptr;
        }
}


cubeNode::~cubeNode() {

    // Delete all m_children
    for (int i = 0; i < 8; ++i) {
        if (m_children[i] != nullptr) {
            delete m_children[i]; // Recursively delete
            m_children[i] = nullptr; // Avoid dangling pointer
        }
    }
}


int cubeNode::addChild(Node* child) {
        for (int i = 0; i < 8; ++i) {
            if (m_children[i] == nullptr) {
                m_children[i] = child;
                child->setParent(this);
                return true;
            }
        }
        return false;
}

//Create new child level of internal cubeNodes, with given data of cubes
void cubeNode::createNewChildLevel(const std::array<Cube, 8>& cubes)
{
    // Iterate through the array of cubes, create 8 new child nodes
        for (int i = 0; i < 8; ++i) {
            // Create a new cubeNode for each m_cube in the array
            this->m_children[i] = new cubeNode(this, cubes[i]); // `this` is the m_parent of the new node
        }
}


// Getter and Setter
Node* cubeNode::getChild(int index) const {
    if (index >= 0 && index < 8) {
        return m_children[index];
    }
    return nullptr;
}

void cubeNode::setChild(int index, Node* child) {
    if (index >= 0 && index < 8) {
        m_children[index] = child;
    }
}
