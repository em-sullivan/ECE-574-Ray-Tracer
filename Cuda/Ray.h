/*
 * Ray Class
 */

#ifndef RAY_H
#define RAY_H

#include "Vec3.h"

class Ray {

public:

    // Constructors
    __device__ Ray();
    __device__ Ray(const Point3 &origin, const Point3 &direction, float time = 0.0f);

    // Getters
    __device__ Point3 origin() const;
    __device__ Vec3 direction() const;
    __device__ float time() const;
    __device__ Point3 at(float t) const;

private:

    Point3 orig;
    Vec3 dir;
    float tm;
};

__device__ Ray::Ray()
{
    orig = Vec3(0, 0,0);
    dir = Vec3(0, 0,0);
    tm = 0.0f;
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

#endif // RAY_H