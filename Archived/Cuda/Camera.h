/*
 * Camera Object
 */

#ifndef CAMERA_H
#define CAMERA_H

#include "Ray.h"
#include "Vec3.h"
#include <curand_kernel.h>
#include "shader_consts.h"

class Camera
{
public:
    __device__ Camera(Point3 lookfrom, Point3 lookat, Vec3 vup, float vfov, float aspect_ratio,
        float aperture, float focus_dist, float _time0 = 0.f, float _time1 = 0.f);
    __device__ Ray get_ray(float u, float v, curandState *local_rand_state);

private:
    Point3 origin;
    Point3 lower_left_corner;
    Vec3 horizontal;
    Vec3 vertical;
    Vec3 u, v, w;
    float lens_radius;
    
    // Shutter open/close times
    float time0;
    float time1;
};

#endif // CAMERA_H