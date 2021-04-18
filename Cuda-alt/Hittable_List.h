/*
 * Class for lists of hittable objects
 */

#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include <memory>
#include "Hittable.h"
#include "Aabb.h"

class Hittable_List : public Hittable
{
public:
    // Constructors
    __device__ Hittable_List();
    __device__ Hittable_List(Hittable **object, int n);

    //void clear();
    //void add(Hittable *object);

    __device__ virtual bool hit(const Ray &r, float t_min, float t_max, hit_record &rec) const override;
    __device__ virtual bool bounding_box(float time0, float time1, Aabb &output_box) const override;

//private:
    Hittable **objects;
    int list_size;
};

__device__ Hittable_List::Hittable_List()
{
    return;
}

__device__ Hittable_List::Hittable_List(Hittable **object, int n)
{
    objects = object;
    list_size = n;
}

__device__ bool Hittable_List::hit(const Ray &r, float t_min, float t_max, hit_record &rec) const
{
    hit_record temp;
    bool hit_any = false;
    float closest = t_max;

    for (int i = 0; i < list_size; i++) {
        if (objects[i]->hit(r, t_min, closest, temp)) {
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
    if (list_size < 1) return false;

    Aabb temp;

    bool first_box =  objects[0]->bounding_box(time0, time1, temp);
    
    if (!first_box) {
        return false;
    } else {
        output_box = temp;
    }
    for (int i = 1; i < list_size; i++) {
        if(objects[i]->bounding_box(time0, time1, temp)) {
            output_box = surrounding_box(output_box, temp);
        }
        else
            return false;
    }
    return true;
}

#endif // HITTABLE_LIST_H