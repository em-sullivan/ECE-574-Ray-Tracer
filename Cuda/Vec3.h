/*
 * Vector class for storing gemometric
 * vecotrs and colors. 
 * Many systesm use 4D vectors, but 3D
 * is what is shown in this tutoiral.
 */

#ifndef VEC3_H
#define VEC3_H

#include <cmath>
#include <iostream>

#include "shader_consts.h"

class Vec3 
{
public:

    // Constructors
    __host__ __device__ Vec3();
    __host__ __device__ Vec3(float e0, float e1, float e2);
    
    // Getters
    __host__ __device__ float x() const;
    __host__ __device__ float y() const;
    __host__ __device__ float z() const;

    // Operators (Overloaded)
    __host__ __device__  Vec3 operator-() const;
    __host__ __device__ float operator[](int i) const;
    __host__ __device__ float& operator[](int i);
    __host__ __device__ Vec3& operator+=(const Vec3 &v);
    __host__ __device__ Vec3& operator-=(const Vec3 &v);
    __host__ __device__ Vec3& operator*=(const float t);
    __host__ __device__ Vec3& operator/=(const float t);

    __host__ __device__ float length() const;
    __host__ __device__ float lengthSquared() const;
    __host__ __device__ bool nearZero() const;

    __device__ inline static Vec3 random(curandState *randState) {
        return Vec3(random_float(randState), random_float(randState), random_float(randState));
    }

    __device__ inline static Vec3 random(float min, float max, curandState *randState) {
        return Vec3(random_float(min, max, randState), random_float(min, max, randState), random_float(min, max, randState));
    }

private:
    float coords[3];
};

// Utility Functions
 __device__ Vec3 randomUnitVector(curandState *local_rand_state);
 __device__ Vec3 randomInHemisphere(const Vec3 &normal, curandState *local_rand_state);
__host__ __device__ Vec3 reflect(const Vec3 &v, const Vec3 &n);
__host__ __device__ Vec3 refract(const Vec3 &v1, const Vec3 &v2, float eta_over_eta);
__device__ Vec3 randomInUnitDisk(curandState *local_rand_state);

inline std::ostream& operator<<(std::ostream &out, const Vec3 &v)
{
    return out << v.x() << " " << v.y() << " " << v.z();
}

__host__ __device__ inline Vec3 operator+(const Vec3 &v1, const Vec3 &v2)
{
    return Vec3(v1.x() + v2.x(), v1.y() + v2.y(), v1.z() + v2.z());
}

__host__ __device__ inline Vec3 operator-(const Vec3 &v1, const Vec3 &v2)
{
    return Vec3(v1.x() - v2.x(), v1.y() - v2.y(), v1.z() - v2.z());
}

__host__ __device__ inline Vec3 operator*(const Vec3 &v1, const Vec3 &v2)
{
    return Vec3(v1.x() * v2.x(), v1.y() * v2.y(), v1.z() * v2.z());
}

__host__ __device__ inline Vec3 operator*(const float t, const Vec3 &v)
{
    return Vec3(v.x() * t, v.y() * t, v.z() * t); 
}

__host__ __device__ inline Vec3 operator*(const Vec3 &v, const float t)
{
    return t * v; 
}

__host__ __device__ inline Vec3 operator/(const Vec3 &v, float t)
{
    return (1 / t) * v;
}

__host__ __device__ inline float dot(const Vec3 &v1, const Vec3 &v2)
{
    return v1.x() * v2.x()
         + v1.y() * v2.y()
         + v1.z() * v2.z();
}

__host__ __device__ inline Vec3 cross(const Vec3 &v1, const Vec3 &v2)
{
    return Vec3(v1.y() * v2.z() - v1.z() * v2.y(),
        v1.z() * v2.x() - v1.x() * v2.z(),
        v1.x() * v2.y() - v1.y() * v2.x());
}

__host__ __device__ inline Vec3 unitVector(Vec3 v)
{
    return v / v.length();
}

__device__ inline Vec3 randomInUnitSphere(curandState *local_rand_state) {
    Vec3 p;
    do {
        p = 2.0f*Vec3::random(local_rand_state) - Vec3(1,1,1);
    } while (p.lengthSquared() >= 1.0f);
    return p;
}

//Vec3 randomUnitVector();
//Vec3 randomInHemisphere(const Vec3 &normal);
//Vec3 reflect(const Vec3 &v, const Vec3 &n);
//Vec3 refract(const Vec3 &v1, const Vec3 &v2, float eta_over_eta);

// Aliases for Vec3
using Color = Vec3;
using Point3 = Vec3;

#endif // VEC3_H