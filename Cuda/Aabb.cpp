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

__device__ inline float ffmin(float a, float b) { return a < b ? a : b; }

__device__ inline float ffmax(float a, float b) { return a > b ? a : b; }

__device__ bool Aabb::hit(const Ray &r, float t_min, float t_max) const
{
    int i;

    // Loop through 3d Vec
    for (i = 0; i < 3; i++) {

        float t0 = ffmin((min()[i] - r.origin()[i]) / r.direction()[i], (max()[i] - r.origin()[i]) / r.direction()[i]);
        float t1 = ffmax((min()[i] - r.origin()[i]) / r.direction()[i], (max()[i] - r.origin()[i]) / r.direction()[i]);
        
        t_min = ffmax(t0, t_min);
        t_max = ffmin(t1, t_max);
        if (t_max <= t_min) {
                    return false;
        }
    }
    return true;
}

__device__ Aabb surrounding_box(Aabb box0, Aabb box1)
{
    Point3 small(ffmin(box0.min().x(), box1.min().x()),
                 ffmin(box0.min().y(), box1.min().y()),
                 ffmin(box0.min().z(), box1.min().z()));

    Point3 big(ffmax(box0.max().x(), box1.max().x()),
               ffmax(box0.max().y(), box1.max().y()),
               ffmax(box0.max().z(), box1.max().z()));

    return Aabb(small, big);
}
