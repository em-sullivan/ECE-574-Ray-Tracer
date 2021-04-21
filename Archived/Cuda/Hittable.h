/*
 * Abstract class - Anything a Ray might Hit
 */

#ifndef HITTABLE_H
#define HITTABLE_H

#include "Ray.h"
#include "shader_consts.h"
#include "Material.h"
#include "Aabb.h"

class Material;

struct hit_record {
    Point3 p;
    Vec3 normal;
    Material *mat_ptr;
    float t;
    float u;
    float v;
    bool front_face;

    __device__ inline void set_face_normal(const Ray &r, const Vec3 &outward_normal)
    {
        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

class Hittable
{
public:
    __device__ virtual bool hit(const Ray &r, float t_min, float t_mix, hit_record &rec) const = 0;
    __device__ virtual bool bounding_box(float time0, float time1, Aabb &output_box) const = 0;
};

#endif // HITTABLE_H