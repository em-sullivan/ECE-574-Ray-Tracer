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
    return (fabsf(coords[0]) < s) && fabsf(coords[1] < s) && fabsf(coords[2] < s);
}
/*
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
*/

/*
__device__ Vec3 randomInUnitDisk(curandState *local_rand_state)
{
    while (true) {
        auto p = Vec3(random_float(local_rand_state), random_float(local_rand_state), 0);
        if (p.lengthSquared() >= 1) continue;
        return p;
   }
}

__device__ Vec3 randomInUnitSphere(curandState *local_rand_state) {
    while (true) {
        auto p = Vec3::random(local_rand_state)-Vec3(1,1,1);
        if (p.lengthSquared() >= 1) continue;
        return p;
    }
}
*/
    __host__ __device__ Vec3& Vec3::operator*=(const Vec3 &v)
    {
    coords[0]  *= v.coords[0];
    coords[1]  *= v.coords[1];
    coords[2]  *= v.coords[2];
    return *this;
}