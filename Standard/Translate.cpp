/*
 * Translate Class source file
 */

#include "Translate.h"

bool Translate::hit(const Ray& r, float t_min, float t_max, hit_record& rec) const 
{
    Ray moved_r(r.origin() - offset, r.direction(), r.time());
    if (!ptr->hit(moved_r, t_min, t_max, rec)) {
        return false;
    }

    rec.p += offset;
    rec.set_face_normal(moved_r, rec.normal);

    return true;
}

bool Translate::bounding_box(float time0, float time1, Aabb& output_box) const
{
    if (!ptr->bounding_box(time0, time1, output_box)) {
        return false;
    }

    output_box = Aabb(output_box.min() + offset,output_box.max() + offset);

    return true;
}

Rotate_Y::Rotate_Y(shared_ptr<Hittable> p, float angle) : ptr(p)
{
    auto radians = deg_to_rad(angle);
    sin_theta = sin(radians);
    cos_theta = cos(radians);
    hasbox = ptr->bounding_box(0, 1, bbox);

    Point3 min( INF,  INF, INF);
    Point3 max(-INF, -INF, -INF);

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
                auto x = i*bbox.max().x() + (1-i)*bbox.min().x();
                auto y = j*bbox.max().y() + (1-j)*bbox.min().y();
                auto z = k*bbox.max().z() + (1-k)*bbox.min().z();

                auto newx =  cos_theta*x + sin_theta*z;
                auto newz = -sin_theta*x + cos_theta*z;

                Vec3 tester(newx, y, newz);

                for (int c = 0; c < 3; c++) {
                    min[c] = fmin(min[c], tester[c]);
                    max[c] = fmax(max[c], tester[c]);
                }
            }
        }
    }

    bbox = Aabb(min, max);
}

bool Rotate_Y::hit(const Ray& r, float t_min, float t_max, hit_record& rec) const
{
    auto origin = r.origin();
    auto direction = r.direction();

    origin[0] = cos_theta*r.origin()[0] - sin_theta*r.origin()[2];
    origin[2] = sin_theta*r.origin()[0] + cos_theta*r.origin()[2];

    direction[0] = cos_theta*r.direction()[0] - sin_theta*r.direction()[2];
    direction[2] = sin_theta*r.direction()[0] + cos_theta*r.direction()[2];

    Ray rotated_r(origin, direction, r.time());

    if (!ptr->hit(rotated_r, t_min, t_max, rec))
        return false;

    auto p = rec.p;
    auto normal = rec.normal;

    p[0] =  cos_theta*rec.p[0] + sin_theta*rec.p[2];
    p[2] = -sin_theta*rec.p[0] + cos_theta*rec.p[2];

    normal[0] =  cos_theta*rec.normal[0] + sin_theta*rec.normal[2];
    normal[2] = -sin_theta*rec.normal[0] + cos_theta*rec.normal[2];

    rec.p = p;
    rec.set_face_normal(rotated_r, normal);

    return true;
}

bool Rotate_Y::bounding_box(float time0, float time1, Aabb& output_box) const
{
    output_box = bbox;
    return hasbox;
}