/*
 * Ray Source File
 */

#include "Vec3.h"
#include "Ray.h"

Ray::Ray()
{
    orig = Vec3(0, 0,0);
    dir = Vec3(0, 0,0);
}

Ray::Ray(const Point3 &origin, const Vec3 &direction)
{
    orig = origin;
    dir = direction;
}

Point3 Ray::origin() const
{
    return orig;
}

Vec3 Ray::direction() const
{
    return dir;
}

Point3 Ray::at(double t) const
{
    //Performs Ray Calculation 
    return orig + t * dir;
}