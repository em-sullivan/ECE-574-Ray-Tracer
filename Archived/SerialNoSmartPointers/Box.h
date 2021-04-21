/*
 * Box Class header file
 */

#ifndef BOX_H
#define BOX_H

#include "Vec3.h"
#include "Ray.h"
#include "shader_consts.h"
#include "Aarect.h"
#include "Hittable.h"
#include "Hittable_List.h"

class Box : public Hittable 
{
    public:
        Box() {}
        Box(const Point3& p0, const Point3& p1, Material *ptr);

        virtual bool hit(const Ray& r, float t_min, float t_max, hit_record& rec) const override;
        virtual bool bounding_box(float time0, float time1, Aabb& output_box) const override;

    private:
        Point3 box_min;
        Point3 box_max;
        Hittable *list_ptr;
};

#endif
