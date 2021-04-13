/*
 * Sphere Class
 */

#ifndef SPHERE_H
#define SPHERE_H

#include "Hittable.h"
#include "Vec3.h"
#include "Material.h"
#include "Texture.h"

class Sphere : public Hittable 
{
public:
    // Constructors
    __device__ Sphere();
     __device__ Sphere(Point3 cen, float r, Material *m);

     __device__ virtual bool hit(const Ray &r, float t_min, float t_max, hit_record &rec) const override;
     __device__ virtual bool bounding_box(float time0, float time1, Aabb &output_box) const override;

//private:
    Point3 center;
    float radius;
    Material *mat_ptr;
};

#endif // SPHERE_H