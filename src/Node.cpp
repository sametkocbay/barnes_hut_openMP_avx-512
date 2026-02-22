#include "Node.h"
#include "cubeNode.h"
#include "particleNode.h"

// Constructor definition
Node::Node(Node* parent, Cube cube) : m_parent(parent), m_cube(cube) {}
// Destructor definition
Node::~Node() {}


// insert method
void Node::insert(Node* parent) {
    //If tree empty: Node must be set to root.

    if(!parent->isParticleNode()) {

        auto cubeParent = dynamic_cast<cubeNode*>(parent);

        // Assign this node as a child of the m_parent cubeNode
            for (int i = 0; i < 8; ++i) {
                if (cubeParent->getChild(i) == nullptr) {
                    cubeParent->setChild(i, this); // Place the current node in the first available slot
                    this->m_parent = parent;             // Set the m_parent for this node
                    return;                            // Exit once inserted
                }
            }

        // If we reach here, there was no space in the m_parent's m_children array
            throw std::overflow_error("Parent cubeNode has no available child slots.");
    } else {
        throw std::invalid_argument("Given m_parent Node is not an internal node");
    }
}


// Getter and setter for m_parent
Node* Node::getParent() const { return m_parent; }
void Node::setParent(Node* p) { m_parent = p; }

// Getter and Setter for m_cube
const Cube &Node::getCube() const { return m_cube; }
void Node::setCube(const Cube &n_cube) { Node::m_cube = n_cube; }