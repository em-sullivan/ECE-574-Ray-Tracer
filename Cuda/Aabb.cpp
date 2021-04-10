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

__device__ bool Aabb::hit(const Ray &r, float t_min, float t_max) const
{
    int i;

    // Loop through 3d Vec
    for (i = 0; i < 3; i++) {

        // Calculate t invtervals over slab
        auto inv_d = 1.0f / r.direction()[i];
        auto t0 = (min()[i] - r.origin()[i]) * inv_d;
        auto t1 = (max()[i] - r.origin()[i]) * inv_d;
        if (inv_d < 0.0f)
            std::swap(t0, t1);

        // This is to deal with overlaping vars or NaNs
        //t_min = fmax(t0, t_min);
        //t_max = fmin(t1, t_max);
        t_min = t0 > t_min ? t0 : t_min;
        t_max = t1 < t_max ? t1 : t_max;
        if (t_max <= t_min)
            return false;
    }

    return true;
}

__device__ Aabb surrounding_box(Aabb box0, Aabb box1)
{
    Point3 small(fmin(box0.min().x(), box1.min().x()),
                 fmin(box0.min().y(), box1.min().y()),
                 fmin(box0.min().z(), box1.min().z()));

    Point3 big(fmax(box0.max().x(), box1.max().x()),
               fmax(box0.max().y(), box1.max().y()),
               fmax(box0.max().z(), box1.max().z()));

    return Aabb(small, big);
}