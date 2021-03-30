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
    Vec3();
    Vec3(float e0, float e1, float e2);
    
    // Getters
    float x() const;
    float y() const;
    float z() const;

    // Operators (Overloaded)
    Vec3 operator-() const;
    float operator[](int i) const;
    float& operator[](int i);
    Vec3& operator+=(const Vec3 &v);
    Vec3& operator-=(const Vec3 &v);
    Vec3& operator*=(const float t);
    Vec3& operator/=(const float t);

    float length() const;
    float lengthSquared() const;
    bool nearZero() const;

    inline static Vec3 random() {
        return Vec3(random_float(), random_float(), random_float());
    }

    inline static Vec3 random(float min, float max) {
        return Vec3(random_float(min, max), random_float(min, max), random_float(min, max));
    }

private:
    float coords[3];
};

// Utility Functions
Vec3 randomUnitVector();
Vec3 randomInHemisphere(const Vec3 &normal);
Vec3 reflect(const Vec3 &v, const Vec3 &n);
Vec3 refract(const Vec3 &v1, const Vec3 &v2, float eta_over_eta);
Vec3 randomInUnitDisk();

inline std::ostream& operator<<(std::ostream &out, const Vec3 &v)
{
    return out << v.x() << " " << v.y() << " " << v.z();
}

inline Vec3 operator+(const Vec3 &v1, const Vec3 &v2)
{
    return Vec3(v1.x() + v2.x(), v1.y() + v2.y(), v1.z() + v2.z());
}

inline Vec3 operator-(const Vec3 &v1, const Vec3 &v2)
{
    return Vec3(v1.x() - v2.x(), v1.y() - v2.y(), v1.z() - v2.z());
}

inline Vec3 operator*(const Vec3 &v1, const Vec3 &v2)
{
    return Vec3(v1.x() * v2.x(), v1.y() * v2.y(), v1.z() * v2.z());
}

inline Vec3 operator*(const float t, const Vec3 &v)
{
    return Vec3(v.x() * t, v.y() * t, v.z() * t); 
}

inline Vec3 operator*(const Vec3 &v, const float t)
{
    return t * v; 
}

inline Vec3 operator/(const Vec3 &v, float t)
{
    return (1 / t) * v;
}

inline float dot(const Vec3 &v1, const Vec3 &v2)
{
    return v1.x() * v2.x()
         + v1.y() * v2.y()
         + v1.z() * v2.z();
}

inline Vec3 cross(const Vec3 &v1, const Vec3 &v2)
{
    return Vec3(v1.y() * v2.z() - v1.z() * v2.y(),
        v1.z() * v2.x() - v1.x() * v2.z(),
        v1.x() * v2.y() - v1.y() * v2.x());
}

inline Vec3 unitVector(Vec3 v)
{
    return v / v.length();
}

inline Vec3 randomInUnitSphere()
{
     while (true) {
        auto p = Vec3::random(-1, 1);
        if (p.lengthSquared() >= 1) continue;
        return p;
    }
}

//Vec3 randomUnitVector();
//Vec3 randomInHemisphere(const Vec3 &normal);
//Vec3 reflect(const Vec3 &v, const Vec3 &n);
//Vec3 refract(const Vec3 &v1, const Vec3 &v2, float eta_over_eta);

// Aliases for Vec3
using Color = Vec3;
using Point3 = Vec3;

#endif // VEC3_H