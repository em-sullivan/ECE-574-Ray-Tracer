/*
 * Source file for BvH_Node class
 */

#include "Bvh_Node.h"
#include <algorithm>

Bvh_Node::Bvh_Node()
{
    // :_
}

Bvh_Node::Bvh_Node(const std::vector<shared_ptr<Hittable>>& src_objects, 
    size_t start, size_t end, float time0, float time1)
{
    auto objects = src_objects;

    int axis = random_int(0, 2);
    auto comparator = (axis == 0) ? box_x_compare
                    : (axis == 1) ? box_y_compare
                                  : box_z_compare;

    size_t object_span = end - start;

    if (object_span == 1) {
        left = right = objects[start];
    } else if (object_span == 2) {
        if (comparator(objects[start], objects[start + 1])) {
            left = objects[start];
            right = objects[start + 1];
        } else {
            left = objects[start + 1];
            right = objects[start];
        }
    } else {

        std::sort(objects.begin() + start, objects.begin() + end, comparator);

        auto mid = start + object_span / 2;
        left = make_shared<Bvh_Node>(objects, start, mid, time0, time1);
        right = make_shared<Bvh_Node>(objects, mid, end, time0, time1);
    }

    Aabb box_left, box_right;

    if (!left->bounding_box(time0, time1, box_left) || !right->bounding_box(time0, time1, box_right))
        std::cerr << "No Bounding box in Bvh_Node constructor" << std::endl;

    box = surrounding_box(box_left, box_right); 
}


bool Bvh_Node::hit(const Ray &r, float t_min, float t_max, hit_record &rec) const
{
    if (!box.hit(r, t_min, t_max))
        return false;

    // Check if either left or right boxes are hit
    bool hit_left = left->hit(r, t_min, t_max, rec);
    bool hit_right = right->hit(r, t_min, hit_left ? rec.t : t_max, rec);

    return hit_left || hit_right;
}

bool Bvh_Node::bounding_box(float time0, float time1, Aabb &output_box) const
{
    output_box = box;
    return true;
}

bool box_x_compare(const shared_ptr<Hittable> a, const shared_ptr<Hittable> b)
{
    return box_compare(a, b, 0);
}

bool box_y_compare(const shared_ptr<Hittable> a, const shared_ptr<Hittable> b)
{
    return box_compare(a, b, 1);
}

bool box_z_compare(const shared_ptr<Hittable> a, const shared_ptr<Hittable> b)
{
    return box_compare(a, b, 2);
}