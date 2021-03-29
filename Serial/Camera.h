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
    Camera(Point3 lookfrom, Point3 lookat, Vec3 vup, double vfov, double aspect_ratio,
        double aperture, double font_dist);
    Ray get_ray(double u, double v);

private:
    Point3 origin;
    Point3 lower_left_corner;
    Vec3 horizontal;
    Vec3 vertical;
    Vec3 u, v, w;
    double lens_radius;
};

#endif // CAMERA_H