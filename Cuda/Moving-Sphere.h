/*
 * A ball the moves (MOITON BLURRRR)
 */

#ifndef MOVING_SPHERE_H
#define MOVING_SPHERE_H

#include "Vec3.h"
#include "Ray.h"
#include "Hittable.h"
//#include "Material.h"
#include "Aabb.h"
#include "shader_consts.h"

class Moving_Sphere : public Hittable 
{
public:

    // Constructors
    __device__ Moving_Sphere();
    __device__ Moving_Sphere(
        Point3 center0,
        Point3 center1,
        float _time0,
        float _time1,
        float r,
        Material *m
    );

    __device__ virtual bool hit(const Ray &r, float t_min, float t_max, hit_record &rec) const override;
    __device__ virtual bool bounding_box(float time0, float time1, Aabb &output_box) const override;
    __device__ Point3 center(float time) const;

private:
    Point3 cen0;
    Point3 cen1;
    float time0;
    float time1;
    float radius;
    //Material *mat_ptr;

};

__device__ Moving_Sphere::Moving_Sphere()
{
    // I don't do anything :(   
}

__device__ Moving_Sphere::Moving_Sphere(Point3 center0, Point3 center1, float _time0, 
    float _time1, float r, Material *m)
{
    cen0 = center0;
    cen1 = center1;
    time0 = _time0;
    time1 = _time1;
    radius = r;
    mat_ptr = m;
}

__device__ bool Moving_Sphere::hit(const Ray &r, float t_min, float t_max, hit_record &rec) const
{
    Vec3 oc = r.origin() - center(r.time());
    auto a = r.direction().lengthSquared();
    auto half_b = dot(oc, r.direction());
    auto c = oc.lengthSquared() - radius * radius;

    auto discriminant = half_b * half_b - a * c;
    if (discriminant < 0) return false;
    auto sqrtd = sqrt(discriminant);

    // Find the nearest root that lies in the acceptable range
    auto root = (-half_b - sqrtd) / a;
    if (root < t_min || root > t_max) {
        root = (-half_b + sqrtd) / a;
        if (root < t_min || root > t_max) {
            return false;
        }
    }

    rec.t = root;
    rec.p = r.at(rec.t);
    Vec3 outward_normal = (rec.p - center(r.time())) / radius;
    rec.set_face_normal(r, outward_normal);
    //rec.normal = (rec.p - center) / radius;
    rec.mat_ptr = mat_ptr;
    return true;
}

__device__ bool Moving_Sphere::bounding_box(float t0, float t1, Aabb &output_box) const
{
    Aabb box0(
        center(t0) - Vec3(radius, radius, radius),
        center(t0) + Vec3(radius, radius, radius)
    );

    Aabb box1(
        center(t1) - Vec3(radius, radius, radius),
        center(t1) + Vec3(radius, radius, radius)
    );

    output_box = surrounding_box(box0, box1);
    return true;
}

__device__ Point3 Moving_Sphere::center(float time) const
{
    return cen0 + ((time - time0) / (time1 - time0)) * (cen1 - cen0);
}

#endif