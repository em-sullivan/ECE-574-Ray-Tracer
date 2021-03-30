/*
 * Class for lists of hittable objects
 */

#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include <memory>
#include <vector>
#include "Hittable.h"

using std::shared_ptr;
using std::make_shared;

class Hittable_List : public Hittable
{
public:
    // Constructors
    Hittable_List();
    Hittable_List(shared_ptr<Hittable> object);

    void clear();
    void add(shared_ptr<Hittable> object);

    virtual bool hit(const Ray &r, float t_min, float t_max, hit_record &rec) const override;

private:
    std::vector<shared_ptr<Hittable>> objects;
};

#endif // HITTABLE_LIST_H