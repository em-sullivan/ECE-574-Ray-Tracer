/*
 * BVH_Node class header file.
 * A container for a hittable_list. Can querey
 * "Does this ray hit you?"
 */

#ifndef BVH_H
#define BVH_H

#include "Ray.h"
#include "Hittable.h"
#include "Hittable_List.h"
#include "Aabb.h"
#include "shader_consts.h"


class Bvh_Node : public Hittable 
{
public:
    
    // Constructors
    Bvh_Node();
    // Need to convert to list format instread of vector
    //Bvh_Node(const Hittable_List &list, float time0, float time1) : Bvh_Node(list.objects, 0, list.objects.size(), time0, time1) {};
    
    //Bvh_Node(const Hittable_List &list, float time0, float time1)
    //        : Bvh_Node(list.objects, 0, list.objects.size(), time0, time1)
    //    {}
    Bvh_Node(Hittable **objects, size_t start, size_t end, float time0, float time1);

    virtual bool hit(const Ray &r, float t_min, float t_mix, hit_record &rec) const override;
    virtual bool bounding_box(float time0, float time1, Aabb &output_box) const override;

private:
    Hittable *left;
    Hittable *right;
    Aabb box;

};

// Utility box compare functions
inline bool box_compare(const Hittable *a, const Hittable *b, int axis)
{
    Aabb box_a, box_b;

    //if (!a->bounding_box(0, 0, box_a) || !b->bounding_box(0, 0, box_b))
        //std::cerr << "No boudning in Bvh_Node constructor " << std::endl;

    return box_a.min()[axis] < box_b.min()[axis];
}

bool box_x_compare(const Hittable *a, const Hittable *b);
bool box_y_compare(const Hittable *a, const Hittable *b);
bool box_z_compare(const Hittable *a, const Hittable *b);
#endif // BVH_H