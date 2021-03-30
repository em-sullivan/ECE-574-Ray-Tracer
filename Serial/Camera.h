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
    Camera(Point3 lookfrom, Point3 lookat, Vec3 vup, float vfov, float aspect_ratio,
        float aperture, float font_dist);
    Ray get_ray(float u, float v);

private:
    Point3 origin;
    Point3 lower_left_corner;
    Vec3 horizontal;
    Vec3 vertical;
    Vec3 u, v, w;
    float lens_radius;
};

#endif // CAMERA_H