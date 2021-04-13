/*
 * Axis Aligned Box Class header file
 */

#ifndef AABB_H
#define AABB_H

#include "Vec3.h"
#include "Ray.h"
#include "shader_consts.h"
#include <thrust/swap.h>

class Aabb
{
public:
    // Constructors
    __device__ Aabb();
    __device__ Aabb(const Point3 &a, const Point3 &b);

    // Getters
    __device__ Point3 max() const;
    __device__ Point3 min() const;

    __device__ bool hit(const Ray &r, float t_min, float t_max) const;

private:

    Point3 maximum;
    Point3 minimum;

};

// Utility Function
__device__ Aabb surrounding_box(Aabb box0, Aabb box1);

#endif //AABB_H