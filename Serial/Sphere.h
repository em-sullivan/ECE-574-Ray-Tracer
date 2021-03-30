/*
 * Sphere Class
 */

#ifndef SPHERE_H
#define SPHERE_H

#include "Hittable.h"
#include "Vec3.h"
#include "Material.h"
#include "shader_consts.h"

class Sphere : public Hittable 
{
public:
    // Constructors
    Sphere();
    Sphere(Point3 cen, float r, shared_ptr<Material> m);

    virtual bool hit(const Ray &r, float t_min, float t_max, hit_record &rec) const override;

private:
    Point3 center;
    float radius;
    shared_ptr<Material> mat_ptr;

};

#endif // SPHERE_H