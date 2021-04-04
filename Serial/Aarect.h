/*
 * Axis Aligned Rectangle Class header file
 */

#ifndef AARECT_H
#define AARECT_H

#include "Vec3.h"
#include "Ray.h"
#include "shader_consts.h"
#include "Hittable.h"

class xy_rect : public Hittable
{
public:
     xy_rect() {}

    xy_rect(float _x0, float _x1, float _y0, float _y1, float _k, shared_ptr<Material> mat)
        : x0(_x0), x1(_x1), y0(_y0), y1(_y1), k(_k), mp(mat) {};

    virtual bool hit(const Ray& r, float t_min, float t_max, hit_record& rec) const override;
    virtual bool bounding_box(float time0, float time1, Aabb& output_box) const override;
        
private:
    float x0, x1, y0, y1, k;
    shared_ptr<Material> mp;
};


class xz_rect : public Hittable
{
public:
     xz_rect() {}

    xz_rect(float _x0, float _x1, float _z0, float _z1, float _k, shared_ptr<Material> mat)
        : x0(_x0), x1(_x1), z0(_z0), z1(_z1), k(_k), mp(mat) {};

    virtual bool hit(const Ray& r, float t_min, float t_max, hit_record& rec) const override;
    virtual bool bounding_box(float time0, float time1, Aabb& output_box) const override;
        
private:
    float x0, x1, z0, z1, k;
    shared_ptr<Material> mp;
};

class yz_rect : public Hittable
{
public:
     yz_rect() {}

    yz_rect(float _y0, float _y1, float _z0, float _z1, float _k, shared_ptr<Material> mat)
        : y0(_y0), y1(_y1), z0(_z0), z1(_z1), k(_k), mp(mat) {};

    virtual bool hit(const Ray& r, float t_min, float t_max, hit_record& rec) const override;
    virtual bool bounding_box(float time0, float time1, Aabb& output_box) const override;
        
private:
    float y0, y1, z0, z1, k;
    shared_ptr<Material> mp;
};

#endif
