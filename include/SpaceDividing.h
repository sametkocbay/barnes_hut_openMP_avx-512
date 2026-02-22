#ifndef BARNESHUT_SPACEDIVIDING_H
#define BARNESHUT_SPACEDIVIDING_H

#include <vector>
#include "ParticleData.h"

class SpaceDividing {

private:
    // root node for all space dividing options?

public:
    virtual Node* divideSpaceSOA(const ParticleData &particles) = 0;
};

#endif //BARNESHUT_SPACEDIVIDING_H
