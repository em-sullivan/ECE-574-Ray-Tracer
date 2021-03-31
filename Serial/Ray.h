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
    Ray(const Point3 &origin, const Point3 &direction, float time = 0.0);

    // Getters
    Point3 origin() const;
    Vec3 direction() const;
    float time() const;


    /*
     * P(t) = A + t *b
     * 
     * P = 3D position along a line
     * A = Origin
     * B = Direction
     * t = Ray
     */
    Point3 at(float t) const;

private:

    Point3 orig;
    Vec3 dir;
    float tm;
};

#endif // RAY_H