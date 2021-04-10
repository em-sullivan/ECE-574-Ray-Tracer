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
    Hittable_List();
    Hittable_List(Hittable *object);

    void clear();
    void add(Hittable *object);

    virtual bool hit(const Ray &r, float t_min, float t_max, hit_record &rec) const override;
    virtual bool bounding_box(float time0, float time1, Aabb &output_box) const override;

//private:
    std::vector<Hittable *> objects;
};

#endif // HITTABLE_LIST_H