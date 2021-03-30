/*
 * Source File for Hittable List
 */

#include "Hittable.h"
#include "Hittable_List.h"

Hittable_List::Hittable_List()
{
    return;
}

Hittable_List::Hittable_List(shared_ptr<Hittable> object)
{
    add(object);
}

void Hittable_List::clear()
{
    objects.clear();
}

void Hittable_List::add(shared_ptr<Hittable> object)
{
    objects.push_back(object);
}

bool Hittable_List::hit(const Ray &r, float t_min, float t_max, hit_record &rec) const
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