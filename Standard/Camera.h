/*
 * Camera Object
 */

#ifndef CAMERA_H
#define CAMERA_H

#include "Ray.h"
#include "Vec3.h"
#include "shader_consts.h"

class Camera
{
public:
    Camera();
    Camera(Point3 lookfrom, Point3 lookat, Vec3 vup, float vfov, float aspect_ratio,
        float aperture, float focus_dist, float _time0 = 0, float _time1 = 0);
    Ray get_ray(float u, float v);

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