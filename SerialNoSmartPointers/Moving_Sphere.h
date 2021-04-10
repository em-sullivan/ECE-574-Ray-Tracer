/*
 * A ball the moves (MOITON BLURRRR)
 */

#ifndef MOVING_SPHERE_H
#define MOVING_SPHERE_H

#include "Vec3.h"
#include "Ray.h"
#include "Hittable.h"
#include "Material.h"
#include "Aabb.h"
#include "shader_consts.h"

class Moving_Sphere : public Hittable 
{
public:

    // Constructors
    Moving_Sphere();
    Moving_Sphere(
        Point3 center0,
        Point3 center1,
        float _time0,
        float _time1,
        float r,
        Material *m
    );

    virtual bool hit(const Ray &r, float t_min, float t_max, hit_record &rec) const override;
    virtual bool bounding_box(float time0, float time1, Aabb &output_box) const override;
    Point3 center(float time) const;

private:
    Point3 cen0;
    Point3 cen1;
    float time0;
    float time1;
    float radius;
    Material *mat_ptr;

};

#endif