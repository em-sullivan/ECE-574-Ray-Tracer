/*
 * Ray Class
 */

#ifndef RAY_H
#define RAY_H

#include "Vec3.h"

class Ray {

public:

    // Constructors
    Ray();
    Ray(const Point3 &origin, const Point3 &direction);

    // Getters
    Point3 origin() const;
    Vec3 direction() const;


    /*
     * P(t) = A + t *b
     * 
     * P = 3D position along a line
     * A = Origin
     * B = Direction
     * t = Ray
     */
    Point3 at(double t) const;

private:

    Point3 orig;
    Vec3 dir;
};

#endif // RAY_H