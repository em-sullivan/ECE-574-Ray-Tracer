/*
 * Axis Aligned Box Class header file
 */

#ifndef AABB_H
#define AABB_H

#include "Vec3.h"
#include "Ray.h"
#include "shader_consts.h"
#include <thrust/swap.h>

__forceinline__ __device__ float ffmin(float a, float b) { return a < b ? a : b; }
__forceinline__ __device__ float ffmax(float a, float b) { return a > b ? a : b; }

class Aabb
{
public:
    // Constructors
    __device__ Aabb();
    __device__ Aabb(const Point3 &a, const Point3 &b);

    // Getters
    __device__ Point3 max() const;
    __device__ Point3 min() const;

    __device__ inline bool hit(const Ray &r, float t_min, float t_max) const
{
    for (int a = 0; a < 3; a++) {
      float invD = 1.0f / r.direction()[a];
      float t0 = (min()[a] - r.origin()[a]) * invD;
      float t1 = (max()[a] - r.origin()[a]) * invD;
      if (invD < 0.0f)
        thrust::swap(t0, t1);
      t_min = t0 > t_min ? t0 : t_min;
      t_max = t1 < t_max ? t1 : t_max;
      if (t_max <= t_min)
        return false;
    }
    return true;
  }

private:

    Point3 maximum;
    Point3 minimum;

};

// Utility Function
__forceinline__ __device__ Aabb surrounding_box(Aabb box0, Aabb box1)
{
    Point3 small(ffmin(box0.min().x(), box1.min().x()),
                        ffmin(box0.min().y(), box1.min().y()),
                        ffmin(box0.min().z(), box1.min().z()));

    Point3 big(ffmax(box0.max().x(), box1.max().x()),
                     ffmax(box0.max().y(), box1.max().y()),
                     ffmax(box0.max().z(), box1.max().z()));

    return Aabb(small, big);
}
#endif //AABB_H