/*
 * Source file for vec3 class
 */

#include <iostream>
#include <cmath>
#include "Vec3.h"

using std::sqrt;

Vec3::Vec3()
{
    coords[0] = 0;
    coords[1] = 0;
    coords[2] = 0;
}

Vec3::Vec3(float e0, float e1, float e2)
{
    coords[0] = e0;
    coords[1] = e1;
    coords[2] = e2;
}

float Vec3::x() const
{
    return coords[0];
}

float Vec3::y() const
{
    return coords[1];
}

float Vec3::z() const
{
    return coords[2];
}

Vec3 Vec3::operator-() const
{
    return Vec3(-coords[0], -coords[1], -coords[2]);
}

float Vec3::operator[](int i) const
{
    // If index is out of range, return left
    // or rightmost coordinate
    if (i < 0)
        return coords[0];
    else if (i > 2)
        return coords[2];
    return coords[i];
}

float& Vec3::operator[](int i)
{
    // If index is out of range, return left
    // or rightmost coordinate
    if (i < 0)
        return coords[0];
    else if (i > 2)
        return coords[2];
    return coords[i];
}

Vec3& Vec3::operator+=(const Vec3 &v)
{
    // Add to vectors
    coords[0] += v.coords[0];
    coords[1] += v.coords[1];
    coords[2] += v.coords[2];
    return *this;
}

Vec3& Vec3::operator-=(const Vec3 &v)
{
    coords[0] -= v.coords[0];
    coords[1] -= v.coords[1];
    coords[2] -= v.coords[2];
    return *this;
}

Vec3& Vec3::operator*=(const float t)
{
    // Multiply all elements in a vector by a constant
    coords[0] *= t;
    coords[1] *= t;
    coords[2] *= t;
    return *this;
}

Vec3& Vec3::operator/=(const float t)
{
    // Divie all elements in a vecotr by a constant
    return *this *= 1 / t;
}

float Vec3::length() const
{
    return sqrt(lengthSquared());
}

float Vec3::lengthSquared() const
{
    return coords[0] * coords[0] +
        coords[1] * coords[1] +
        coords[2] * coords[2];
}

bool Vec3::nearZero() const
{
    // Return true if the vector is close to zero in all dimensions
    const float s = 1e-8;
    return (fabs(coords[0]) < s) && fabs(coords[1] < s) && fabs(coords[2] < s);
}

Vec3 randomUnitVector()
{
    return unitVector(randomInUnitSphere());
}

Vec3 randomInHemisphere(const Vec3 &normal)
{
    Vec3 in_unit_sphere = randomInUnitSphere();

    // In the same hemisphere as the normal
    if (dot(in_unit_sphere, normal) > 0.0)
        return in_unit_sphere;
    else
        return -in_unit_sphere;
}

Vec3 reflect(const Vec3 &v, const Vec3 &n)
{
    return v - 2 * dot(v, n) * n;
}

Vec3 refract(const Vec3 &v1, const Vec3 &v2, float etai_over_etat)
{
    float cos_theta = fmin(dot(-v1, v2), 1.0);
    Vec3 r_out_perp = etai_over_etat * (v1 + cos_theta * v2);
    Vec3 r_out_parallel = -sqrt(fabs(1.0 - r_out_perp.lengthSquared())) * v2;
    return r_out_perp + r_out_parallel;
}

Vec3 randomInUnitDisk()
{
    while (true) {
        auto p = Vec3(random_float(-1, 1), random_float(-1, 1), 0);
        if (p.lengthSquared() >= 1) continue;
        return p;
    }
}