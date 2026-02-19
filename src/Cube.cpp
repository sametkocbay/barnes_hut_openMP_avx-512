#include "Cube.h"
//Constructor
Cube::Cube(const std::array<double, 3>& centerPoint, const double norm)
        : centerPoint(centerPoint), norm(norm) {
            this->massCalculated = false;
}
Cube::Cube() {
    this->massCalculated = false;
}

// Setter and Getter Functions

void Cube::setCenterPoint(const std::array<double, 3>& centerPoint) {
    this->centerPoint = centerPoint;
}

const std::array<double, 3>& Cube::getCenterPoint() const {
    return centerPoint;
}

double Cube::getNorm() const {
    return norm;
}

void Cube::setNorm(double norm) {
    this->norm = norm;
}

double Cube::getMass() const {
    return mass;
}

void Cube::setMass(double mass) {
    this->mass = mass;
    this->massCalculated = true;
}

const std::array<double, 3>& Cube::getCenterOfMass() const {
    return centerOfMass;
}

void Cube::setCenterOfMass(const std::array<double, 3>& centerOfMass) {
    this->centerOfMass = centerOfMass;
}

bool Cube::isMassCalculated() const
{
    return massCalculated;
}

