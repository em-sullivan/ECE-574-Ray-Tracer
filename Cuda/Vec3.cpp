/*
 * Source file for vec3 class
 */

#include <iostream>
#include "Vec3.h"

using std::sqrt;

__host__ __device__ Vec3::Vec3()
{
    coords[0] = 0;
    coords[1] = 0;
    coords[2] = 0;
}

__host__ __device__ Vec3::Vec3(float e0, float e1, float e2)
{
    coords[0] = e0;
    coords[1] = e1;
    coords[2] = e2;
}

__host__ __device__ float Vec3::x() const
{
    return coords[0];
}

__host__ __device__ float Vec3::y() const
{
    return coords[1];
}

__host__ __device__ float Vec3::z() const
{
    return coords[2];
}

__host__ __device__ Vec3 Vec3::operator-() const
{
    return Vec3(-coords[0], -coords[1], -coords[2]);
}

__host__ __device__ float Vec3::operator[](int i) const
{
    // If index is out of range, return left
    // or rightmost coordinate
    if (i < 0)
        return coords[0];
    else if (i > 2)
        return coords[2];
    return coords[i];
}

__host__ __device__ float& Vec3::operator[](int i)
{
    // If index is out of range, return left
    // or rightmost coordinate
    if (i < 0)
        return coords[0];
    else if (i > 2)
        return coords[2];
    return coords[i];
}

__host__ __device__ Vec3& Vec3::operator+=(const Vec3 &v)
{
    // Add to vectors
    coords[0] += v.coords[0];
    coords[1] += v.coords[1];
    coords[2] += v.coords[2];
    return *this;
}

__host__ __device__ Vec3& Vec3::operator-=(const Vec3 &v)
{
    coords[0] -= v.coords[0];
    coords[1] -= v.coords[1];
    coords[2] -= v.coords[2];
    return *this;
}

__host__ __device__ Vec3& Vec3::operator*=(const float t)
{
    // Multiply all elements in a vector by a constant
    coords[0] *= t;
    coords[1] *= t;
    coords[2] *= t;
    return *this;
}

__host__ __device__ Vec3& Vec3::operator/=(const float t)
{
    // Divie all elements in a vecotr by a constant
    return *this *= 1 / t;
}

__host__ __device__ float Vec3::length() const
{
    return sqrtf(lengthSquared());
}

__host__ __device__ float Vec3::lengthSquared() const
{
    return coords[0] * coords[0] +
        coords[1] * coords[1] +
        coords[2] * coords[2];
}

__host__ __device__ bool Vec3::nearZero() const
{
    // Return true if the vector is close to zero in all dimensions
    const float s = 1e-8;
    return (fabsf(coords[0]) < s) && fabsf(coords[1] < s) && fabsf(coords[2] < 2);
}

__device__ Vec3 randomUnitVector(curandState *local_rand_state)
{
    return unitVector(randomInUnitSphere(local_rand_state));
}

__device__ Vec3 randomInHemisphere(const Vec3 &normal, curandState *local_rand_state)
{
    Vec3 in_unit_sphere = randomInUnitSphere(local_rand_state);

    // In the same hemisphere as the normal
    if (dot(in_unit_sphere, normal) > 0.0)
        return in_unit_sphere;
    else
        return -in_unit_sphere;
}

__host__ __device__ Vec3 reflect(const Vec3 &v, const Vec3 &n)
{
    return v - 2 * dot(v, n) * n;
}

__host__ __device__ Vec3 refract(const Vec3 &v1, const Vec3 &v2, float etai_over_etat)
{
    float cos_theta = fminf(dot(-v1, v2), 1.0);
    Vec3 r_out_perp = etai_over_etat * (v1 + cos_theta * v2);
    Vec3 r_out_parallel = -sqrtf(fabsf(1.0 - r_out_perp.lengthSquared())) * v2;
    return r_out_perp + r_out_parallel;
}

__device__ Vec3 randomInUnitDisk(curandState *local_rand_state)
{
    Vec3 p;
    do {
        p = 2.0f*Vec3(random_float(local_rand_state),random_float(local_rand_state),0) - Vec3(1,1,0);
    } while (dot(p,p) >= 1.0f);
    return p;
}