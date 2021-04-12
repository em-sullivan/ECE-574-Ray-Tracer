/*
 * Camera Source File
 */

#include "Camera.h"

__device__ Vec3 randomInUnitDisk(curandState *local_rand_state)
{
    Vec3 p;
    do {
        p = 2.0f*Vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), 0) - Vec3(1,1,0);
   } while (dot(p,p) >= 1.0f);
   return p;
}

__device__ Camera::Camera(Point3 lookfrom, Point3 lookat, Vec3 vup, float vfov, float aspect_ratio,
    float aperture, float focus_dist, float _time0, float _time1)
{
    auto theta = deg_to_rad(vfov);
    auto h = tanf(theta / 2);
    auto viewport_height = 2.0 * h;
    auto viewport_width = aspect_ratio * viewport_height;

    w = unitVector(lookfrom - lookat);
    u = unitVector(cross(vup, w));
    v = cross(w, u);

    origin = lookfrom;
    horizontal = focus_dist * viewport_width * u;
    vertical = focus_dist * viewport_height * v;
    lower_left_corner = origin - horizontal / 2 - vertical / 2 - focus_dist * w;

    lens_radius = aperture / 2;
    time0 = _time0;
    time1 = _time1;
}

__device__ Ray Camera::get_ray(float s, float t, curandState *local_rand_state)
{
    Vec3 rd = lens_radius * randomInUnitDisk(local_rand_state);
    Vec3 offset = u * rd.x() + v * rd.y();
    return Ray(origin + offset, lower_left_corner + s * horizontal + t * vertical - origin - offset, random_float(time0, time1, local_rand_state));
}