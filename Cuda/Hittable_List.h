/*
 * Class for lists of hittable objects
 */

#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include <memory>
#include <vector>
#include "Hittable.h"
#include "Aabb.h"

class Hittable_List : public Hittable
{
public:
    // Constructors
    __device__ Hittable_List();
    __device__ Hittable_List(Hittable *object);

    __device__ void clear();
    __device__ void add(Hittable *object);

    __device__ virtual bool hit(const Ray &r, float t_min, float t_max, hit_record &rec) const override;
    __device__ virtual bool bounding_box(float time0, float time1, Aabb &output_box) const override;

//private:
    std::vector<Hittable *> objects;
};

#endif // HITTABLE_LIST_H