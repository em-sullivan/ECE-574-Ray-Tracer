/*
 * Axis Aligned Rectangle Class source file
 */

#include "Aarect.h"

bool xy_rect::hit(const Ray& r, float t_min, float t_max, hit_record& rec) const 
{
    auto t = (k-r.origin().z()) / r.direction().z();
    if (t < t_min || t > t_max) {
        return false;
    }

    auto x = r.origin().x() + t*r.direction().x();
    auto y = r.origin().y() + t*r.direction().y();
    if (x < x0 || x > x1 || y < y0 || y > y1) {
        return false;
    }
    
    rec.u = (x-x0)/(x1-x0);
    rec.v = (y-y0)/(y1-y0);
    rec.t = t;
    auto outward_normal = Vec3(0, 0, 1);
    
    rec.set_face_normal(r, outward_normal);
    rec.mat_ptr = mp;
    rec.p = r.at(t);
    
    return true;
}

bool xy_rect::bounding_box(float time0, float time1, Aabb& output_box) const
{
    // The bounding box must have non-zero width in each dimension, so pad the dimension a small amount.
    output_box = Aabb(Point3(x0, y0, k-0.0001), Point3(x1, y1, k+0.0001));
    return true;
}