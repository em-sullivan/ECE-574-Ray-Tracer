/*
 * Classes for translating and rotating objects
 */

#ifndef TRANSLATE_H
#define TRANSLATE_H

#include "Ray.h"
#include "shader_consts.h"
#include "Material.h"
#include "Aabb.h"

class Translate : public Hittable
{
    public:
        Translate(Hittable *p, const Vec3& displacement)
            : ptr(p), offset(displacement) {}

        virtual bool hit(const Ray& r, float t_min, float t_max, hit_record& rec) const override;
        virtual bool bounding_box(float time0, float time1, Aabb& output_box) const override;

    private:
        Vec3 offset;
        Hittable *ptr;
};

class Rotate_Y : public Hittable
{
    public:
        Rotate_Y(Hittable *p, float angle);

        virtual bool hit(const Ray& r, float t_min, float t_max, hit_record& rec) const override;
        virtual bool bounding_box(float time0, float time1, Aabb& output_box) const override;

    private:
        Hittable *ptr;
        float sin_theta;
        float cos_theta;
        bool hasbox;
        Aabb bbox;
};






#endif