/*
 * BoxClass source file
 */

#include "Box.h"



Box::Box(const Point3& p0, const Point3& p1, Material *ptr)
{
    box_min = p0;
    box_max = p1;

    sides.add( new XY_Rect(p0.x(), p1.x(), p0.y(), p1.y(), p1.z(), ptr));
    sides.add(new XY_Rect(p0.x(), p1.x(), p0.y(), p1.y(), p0.z(), ptr));

    sides.add(new XZ_Rect(p0.x(), p1.x(), p0.z(), p1.z(), p1.y(), ptr));
    sides.add(new XZ_Rect(p0.x(), p1.x(), p0.z(), p1.z(), p0.y(), ptr));

    sides.add(new YZ_Rect(p0.y(), p1.y(), p0.z(), p1.z(), p1.x(), ptr));
    sides.add(new YZ_Rect(p0.y(), p1.y(), p0.z(), p1.z(), p0.x(), ptr));
}

bool Box::hit(const Ray& r, float t_min, float t_max, hit_record& rec) const
{
    return sides.hit(r, t_min, t_max, rec);
}

bool Box::bounding_box(float time0, float time1, Aabb& output_box) const
{
    output_box = Aabb(box_min, box_max);
    return true;
}