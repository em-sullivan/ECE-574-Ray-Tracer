/*
 * Sphere Class
 */

#ifndef SPHERE_H
#define SPHERE_H

#include "Hittable.h"
#include "Vec3.h"
#include "Material.h"
#include "Texture.h"
#include "shader_consts.h"

class Sphere : public Hittable 
{
public:
    // Constructors
    Sphere();
    Sphere(Point3 cen, float r, Material *m);

    virtual bool hit(const Ray &r, float t_min, float t_max, hit_record &rec) const override;
    virtual bool bounding_box(float time0, float time1, Aabb &output_box) const override;

private:
    Point3 center;
    float radius;
    Material *mat_ptr;
    static void get_sphere_uv(Point3 &p, float &u, float &v);

};

#endif // SPHERE_H