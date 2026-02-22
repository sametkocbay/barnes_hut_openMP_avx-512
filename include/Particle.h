#ifndef BARNESHUT_PARTICLE_H
#define BARNESHUT_PARTICLE_H

#include <array>

class Particle {
private:
    // Position
    std::array<double, 3> Position;

    // Velocity
    std::array<double, 3> Velocity;

    // Mass
    double mass;

public:
    // Constructor
    Particle(const std::array<double, 3> &Position, const std::array<double, 3> &Velocity, double mass);

    // Operator overloads
    bool operator==(const Particle& other) const
    {
        return((Position == other.getPosition()) && (Velocity == other.getVelocity()) && (mass == other.getMass()));
    }
    bool operator!=(const Particle& other) const
    {
        return !(*this==other);
    }

    // Getter and Setter for Mass
    double getMass() const;
    void setMass(double mass);

    // Getter and Setter for Velocity
    const std::array<double, 3>& getVelocity() const;
    void setVelocity(const std::array<double, 3>& velocity);

    // Getter and Setter for Position
    const std::array<double, 3>& getPosition() const;
    void setPosition(const std::array<double, 3>& position);
};

#endif //BARNESHUT_PARTICLE_H