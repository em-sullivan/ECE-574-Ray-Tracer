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
    get_sphere_uv(outward_normal, rec.u, rec.v);
    rec.mat_ptr = mat_ptr;
    return true;
}

 __device__ bool Sphere::bounding_box(float time0, float time1, Aabb &output_box) const
{
    output_box = Aabb(
        center - Vec3(radius, radius, radius),
        center + Vec3(radius, radius, radius)
    );

    return true;
}

 __device__ void Sphere::get_sphere_uv(Point3 &p, float &u, float &v)
{
    /*
     * P - A givne point on the sphere of radius one, centered at origin
     * u - return value [0,1] of angle around Y axi from X=-1
     * v - return value [0, 1] of angle from y=-1 tp y=1
     */

    auto theta = acos(-p.y());
    auto phi = atan2(-p.z(), p.x()) + PI;

    u = phi / (2 * PI);
    v = theta / PI;
}