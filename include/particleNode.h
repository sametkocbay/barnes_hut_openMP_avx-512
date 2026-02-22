#ifndef BARNESHUT_PARTICLENODE_H
#define BARNESHUT_PARTICLENODE_H

#include <optional>
#include <vector>
#include "Node.h"  // Include Node.h to inherit from Node
#include "Cube.h"
#include "Particle.h"

class particleNode : public Node {

private:

    std::optional<Particle> m_particle;

public:

    //Constructors
    particleNode(Node *parent, const Cube &cube, const std::optional<Particle>& particle);
    particleNode(Node *parent, const Cube &cube);

    //Destructor
    ~particleNode() override;

    [[nodiscard]] bool isParticleNode() const override { return true; }

    //Setter and Getter
    [[nodiscard]] const std::optional<Particle>& getParticle() const;
    void setParticle(const std::optional<Particle>& particle);

};

#endif //BARNESHUT_PARTICLENODE_H
