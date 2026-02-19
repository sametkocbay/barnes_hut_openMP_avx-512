#include "particleNode.h"
#include "Node.h"
#include <optional>

//Constructor
particleNode::particleNode(Node *parent, const Cube &cube, const std::optional<Particle>& particle) :
        Node(parent,cube), m_particle(particle) {}

particleNode::particleNode(Node *parent, const Cube &cube) :
        Node(parent, cube) {}


//Getter and Setter
const std::optional<Particle>& particleNode::getParticle() const {
    return m_particle;
}

void particleNode::setParticle(const std::optional<Particle>& n_particle) {
    particleNode::m_particle = n_particle;
}


particleNode::~particleNode() {}
