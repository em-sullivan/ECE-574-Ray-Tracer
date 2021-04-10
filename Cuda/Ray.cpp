/*
 * Ray Source File
 */

#include "Vec3.h"
#include "Ray.h"

__device__ Ray::Ray()
{
    orig = Vec3(0, 0,0);
    dir = Vec3(0, 0,0);
    tm = 0.0;
}

__device__ Ray::Ray(const Point3 &origin, const Vec3 &direction, float time)
{
    orig = origin;
    dir = direction;
    tm = time;
}

__device__ Point3 Ray::origin() const
{
    return orig;
}

__device__ Vec3 Ray::direction() const
{
    return dir;
}

__device__ float Ray::time() const
{
    return tm;
}

__device__ Point3 Ray::at(float t) const
{
    //Performs Ray Calculation 
    return orig + t * dir;
}