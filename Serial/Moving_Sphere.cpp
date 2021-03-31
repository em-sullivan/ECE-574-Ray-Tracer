/*
 * Moving Sphere (Balls) source file
 */

#include "Moving_Sphere.h"

Moving_Sphere::Moving_Sphere()
{
    // I don't do anything :(   
}

Moving_Sphere::Moving_Sphere(Point3 center0, Point3 center1, float _time0, 
    float _time1, float r, shared_ptr<Material> m)
{
    cen0 = center0;
    cen1 = center1;
    time0 = _time0;
    time1 = _time1;
    radius = r;
    mat_ptr = m;
}

bool Moving_Sphere::hit(const Ray &r, float t_min, float t_max, hit_record &rec) const
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

Point3 Moving_Sphere::center(float time) const
{
    return cen0 + ((time - time0) / (time1 - time0)) * (cen1 - cen0);
}