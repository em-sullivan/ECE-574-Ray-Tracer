/*
 * Sphere Class
 */

#ifndef SPHERE_H
#define SPHERE_H

#include "Hittable.h"
#include "Vec3.h"
//#include "Material.h"
#include "Texture.h"
#include "shader_consts.h"

__device__ void get_sphere_uv(const Vec3& p, float& u, float& v) {
    float phi = atan2f(p.z(), p.x());
    float theta = asinf(p.y());
    u = 1.f-(phi + PI) / (2.f*PI);
    v = (theta + PI/2.f) / PI;
}

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
    //Material *mat_ptr;
};

__device__ Sphere::Sphere(Point3 cen, float r, Material *m)
{
    center = cen;
    radius = r;
    mat_ptr = m;
}

 __device__ bool Sphere::hit(const Ray &r, float t_min, float t_max, hit_record &rec) const
{
    Vec3 oc = r.origin() - center;
    float a = dot(r.direction(), r.direction());
    float b = dot(oc, r.direction());
    float c = dot(oc, oc) - radius*radius;
    float discriminant = b*b - a*c;
    if (discriminant > 0.f) {
        float temp = (-b - sqrtf(discriminant))/a;
    if (temp < t_max && temp > t_min) {
      rec.t = temp;
      rec.p = r.at(rec.t);
      get_sphere_uv((rec.p-center)/radius, rec.u, rec.v); //new
      rec.normal = (rec.p - center) / radius;
      rec.mat_ptr = mat_ptr;
      return true;
    }
    temp = (-b + sqrtf(discriminant)) / a;
    if (temp < t_max && temp > t_min) {
      rec.t = temp;
      rec.p = r.at(rec.t);
      get_sphere_uv((rec.p-center)/radius, rec.u, rec.v);
      rec.normal = (rec.p - center) / radius;
      rec.mat_ptr = mat_ptr;
      return true;
    }
  }
  return false;
}

 __forceinline__ __device__ bool Sphere::bounding_box(float time0, float time1, Aabb &output_box) const
{
    (void)time0;
    (void)time1;
    output_box = Aabb(center - Vec3(radius, radius, radius), center + Vec3(radius, radius, radius));

    return true;
}


#endif // SPHERE_H