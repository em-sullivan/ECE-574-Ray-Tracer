/*
 * Source file for Sphere class
 */

#include "Sphere.h"


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
      rec.normal = (rec.p - center) / radius;
      rec.mat_ptr = mat_ptr;
      return true;
    }
    temp = (-b + sqrtf(discriminant)) / a;
    if (temp < t_max && temp > t_min) {
      rec.t = temp;
      rec.p = r.at(rec.t);
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
