#ifndef CONSTANT_MEDIUM_H
#define CONSTANT_MEDIUM_H

#include "shader_consts.h"

#include "Hittable.h"
#include "Material.h"
#include "Texture.h"

class Constant_Medium : public Hittable 
{
    public:
        Constant_Medium(Hittable *b, float d, Texture *a) : boundary(b), neg_inv_density(-1/d)
        {
            phase_function = new Isotropic(a);
        }

        Constant_Medium(Hittable *b, float d, Color c) : boundary(b), neg_inv_density(-1/d)
        {
            phase_function = new Isotropic(c);
        }
        
        virtual bool hit(const Ray& r, float t_min, float t_max, hit_record& rec) const override;
        virtual bool bounding_box(float time0, float time1, Aabb& output_box) const override;

        private: 
            Hittable *boundary;
            Material *phase_function;
            float neg_inv_density;
};

#endif