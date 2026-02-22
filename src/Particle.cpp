#include <array>
#include "Particle.h"

// Constructor
Particle::Particle(const std::array<double, 3> &Position, const std::array<double, 3> &Velocity, double mass)
    : Position(Position), Velocity(Velocity), mass(mass) {}

// Getter and Setter for Mass
double Particle::getMass() const {
    return mass;
}

void Particle::setMass(double mass) {
    this->mass = mass;
}

// Getter and Setter for Velocity
const std::array<double, 3>& Particle::getVelocity() const {
    return Velocity;
}

void Particle::setVelocity(const std::array<double, 3>& velocity) {
    Velocity = velocity;
}

// Getter and Setter for Position
const std::array<double, 3>& Particle::getPosition() const {
    return Position;
}

void Particle::setPosition(const std::array<double, 3>& position) {
    Position = position;
}