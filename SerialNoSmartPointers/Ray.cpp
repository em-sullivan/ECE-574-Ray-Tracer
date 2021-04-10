/*
 * Ray Source File
 */

#include "Vec3.h"
#include "Ray.h"

Ray::Ray()
{
    orig = Vec3(0, 0,0);
    dir = Vec3(0, 0,0);
    tm = 0.0;
}

Ray::Ray(const Point3 &origin, const Vec3 &direction, float time)
{
    orig = origin;
    dir = direction;
    tm = time;
}

Point3 Ray::origin() const
{
    return orig;
}

Vec3 Ray::direction() const
{
    return dir;
}

float Ray::time() const
{
    return tm;
}

Point3 Ray::at(float t) const
{
    //Performs Ray Calculation 
    return orig + t * dir;
}