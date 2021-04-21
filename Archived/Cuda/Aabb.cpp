/*
 * Source file for Axis-aligned bounding boax
 */

#include "Aabb.h"

__device__ Aabb::Aabb()
{
    // Nada
}

__device__ Aabb::Aabb(const Point3 &min, const Point3 &max)
{
    maximum = max;
    minimum = min;
}

__device__ Point3 Aabb::max() const
{
    return maximum;
}

__device__ Point3 Aabb::min() const
{
    return minimum;
}
