/*
 * Axis Aligned Box Class header file
 */

#ifndef AABB_H
#define AABB_H

#include "Vec3.h"
#include "Ray.h"
#include "shader_consts.h"

class Aabb
{
public:
    // Constructors
    Aabb();
    Aabb(const Point3 &a, const Point3 &b);

    // Getters
    Point3 max() const;
    Point3 min() const;

    bool hit(const Ray &r, float t_min, float t_max) const;

private:

    Point3 maximum;
    Point3 minimum;

};

// Utility Function
Aabb surrounding_box(Aabb box0, Aabb box1);

#endif //AABB_H