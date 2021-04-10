/*
 * Source File for Hittable List
 */

#include "Hittable.h"
#include "Hittable_List.h"

__device__ Hittable_List::Hittable_List()
{
    return;
}

__device__ Hittable_List::Hittable_List(shared_ptr<Hittable> object)
{
    add(object);
}

__device__ void Hittable_List::clear()
{
    objects.clear();
}

__device__ void Hittable_List::add(shared_ptr<Hittable> object)
{
    objects.push_back(object);
}

__device__ bool Hittable_List::hit(const Ray &r, float t_min, float t_max, hit_record &rec) const
{
    hit_record temp;
    bool hit_any = false;
    float closest = t_max;

    // Loop through the objects in the Hittable List
    for (const auto &object: objects) {

        // Update Closest - if anything is hit
        if (object->hit(r, t_min, closest, temp)) {
            hit_any = true;
            closest = temp.t;
            rec = temp;
        }
    }

    return hit_any;
}

__device__ bool Hittable_List::bounding_box(float time0, float time1, Aabb &output_box) const
{
    // If hitable list is empty
    if (objects.empty()) return false;

    Aabb temp;
    bool first_box = true;

    for (const auto &object : objects) {
        if (!object->bounding_box(time0, time1, temp)) return false;

        output_box = first_box ? temp : surrounding_box(output_box, temp);
        first_box = false;
    }

    return true;
}