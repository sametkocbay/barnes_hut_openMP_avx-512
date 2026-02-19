//
// Created by fabia on 13.11.2024.
//

#ifndef BARNESHUT_CUBE_H
#define BARNESHUT_CUBE_H


#include <array>

class Cube {

private:
    std::array<double, 3> centerPoint; // x, y, z
    std::array<double, 3> centerOfMass; // x, y, z
    double norm; // length to edge
    double mass; // mass
    bool massCalculated;

public:
    //Constrcutor
    // Constructor declaration: Übergibt nur Centerpoint und Norm
    Cube(const std::array<double, 3>& centerPoint, const double norm);
    Cube();

    // Getter and Setter functions
    void setCenterPoint(const std::array<double, 3>& centerPoint);
    const std::array<double, 3>& getCenterPoint() const;

    double getNorm() const;
    void setNorm(double norm);

    double getMass() const;
    void setMass(double mass);

    const std::array<double, 3>& getCenterOfMass() const;
    void setCenterOfMass(const std::array<double, 3>& centerOfMass);

    bool isMassCalculated() const;

};



#endif //BARNESHUT_CUBE_H
