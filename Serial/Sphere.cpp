/*
 * Source file for Sphere class
 */

#include "Sphere.h"


Sphere::Sphere(Point3 cen, float r, shared_ptr<Material> m)
{
    center = cen;
    radius = r;
    mat_ptr = m;
}

bool Sphere::hit(const Ray &r, float t_min, float t_max, hit_record &rec) const
{
    Vec3 oc = r.origin() - center;
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
    Vec3 outward_normal = (rec.p - center) / radius;
    rec.set_face_normal(r, outward_normal);
    //rec.normal = (rec.p - center) / radius;
    rec.mat_ptr = mat_ptr;
    return true;
}