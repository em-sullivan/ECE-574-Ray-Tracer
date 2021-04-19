/*
 * Abstract class - Anything a Ray might Hit
 */

#ifndef HITTABLE_H
#define HITTABLE_H

#include "Ray.h"
#include "shader_consts.h"
//#include "Material.h"
#include "Aabb.h"
#include "Texture.h"

class Material;

struct hit_record {
    Point3 p;
    Vec3 normal;
    Material *mat_ptr;
    float t;
    float u;
    float v;
    bool front_face;

    __device__ inline void set_face_normal(const Ray &r, const Vec3 &outward_normal)
    {
        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

class Hittable
{
public:
    __device__ virtual bool hit(const Ray &r, float t_min, float t_mix, hit_record &rec) const = 0;
    __device__ virtual bool bounding_box(float time0, float time1, Aabb &output_box) const = 0;
};



/*
 * NOTE:
 * Couldn't get the code from material to compile, kept getting a 
 *  "Incomplete Type not allowed", so the material code is included
 * with the hittable code (for now)
 */

#define RANDVEC3 Vec3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))

__device__ Vec3 randomInUnitSphere(curandState *local_rand_state) 
{
    Vec3 p;
    do {
        p = 2.0f*RANDVEC3 - Vec3(1,1,1);
    } while (p.lengthSquared() >= 1.0f);
    return p;
}

__device__ Vec3 randomUnitVector(curandState *local_rand_state)
{
    return unitVector(randomInUnitSphere(local_rand_state));
}

__device__ Vec3 randomInHemisphere(const Vec3 &normal, curandState *local_rand_state)
{
    Vec3 in_unit_sphere = randomInUnitSphere(local_rand_state);

    // In the same hemisphere as the normal
    if (dot(in_unit_sphere, normal) > 0.0f)
        return in_unit_sphere;
    else
        return -in_unit_sphere;
}

__device__ Vec3 reflect(const Vec3 &v, const Vec3 &n)
{
    return v - 2.f*dot(v, n)*n;
}

/*
__device__ Vec3 refract(const Vec3 &v1, const Vec3 &v2, float etai_over_etat)
{
    float cos_theta = fminf(dot(-v1, v2), 1.0f);
    Vec3 r_out_perp = etai_over_etat * (v1 + cos_theta * v2);
    Vec3 r_out_parallel = -sqrtf(fabsf(1.0f - r_out_perp.lengthSquared())) * v2;
    return r_out_perp + r_out_parallel;
}
*/

__device__ bool refract(const Vec3& v, const Vec3& n, float ni_over_nt, Vec3& refracted) {
  Vec3 uv = unitVector(v);
  float dt = dot(uv, n);
  float discriminant = 1.0f - ni_over_nt*ni_over_nt*(1-dt*dt);
  if (discriminant > 0.f) {
    refracted = ni_over_nt*(uv - n*dt) - n*sqrtf(discriminant);
    return true;
  }
  else
    return false;
}

class Material
{
public:
    __device__ virtual bool scatter(const Ray &r_in, hit_record &rec, Color &attenuation, Ray &scattered, curandState *local_rand_state) const
    {
        // Generic implementation - does nothing
        return false;
    }

    __device__ virtual Color emitted(float u, float v, const Point3& p) const
    {
        return Color(0,0,0);
    }

};

class Lambertian : public Material
{
public:
    // Constructor
   /* __device__ Lambertian(const Color &a) 
    {
        albedo = new Solid_Color (a);
    }
*/
    __device__ Lambertian(Texture *a)
    {
        albedo = a;
    }

    __device__ virtual bool scatter(const Ray &r_in, hit_record &rec, Color &attenuation, Ray &scattered, curandState *local_rand_state) const
    {
        Vec3 target = rec.p + rec.normal + randomInUnitSphere(local_rand_state);
        scattered = Ray(rec.p, target-rec.p, r_in.time());
        attenuation = albedo->value(0, 0, rec.p);
        return true;

    }

private:
    //Color albedo;
    Texture *albedo;
};

class Metal : public Material
{
public:
    // Constructor
    __device__ Metal(const Color &a, float in_fuzz)
    {
        albedo = a;
        fuzz = in_fuzz < 1.f ? in_fuzz : 1.f;
    }


    __device__ virtual bool scatter(const Ray &r_in, hit_record &rec, Color &attenuation, Ray &scattered, curandState *local_rand_state) const
    {
        Vec3 reflected = reflect(unitVector(r_in.direction()), rec.normal);
        scattered = Ray(rec.p, reflected + fuzz * randomInUnitSphere(local_rand_state), r_in.time());
        attenuation = albedo;
        return (dot(scattered.direction(), rec.normal) > 0.0f);
    }

private:
    Color albedo;
    float fuzz;
};

class Dielectric : public Material
{
public:
    __device__ Dielectric(float refraction_index)
    {
        refraction = refraction_index;
    }

    __device__ virtual bool scatter(const Ray &r_in, hit_record &rec, Color &attenuation, Ray &scattered, curandState *local_rand_state) const
    {
        Vec3 outward_normal;
    float refraction_ratio = refraction;
    Vec3 reflected = reflect(r_in.direction(), rec.normal); //ONLY REFLECT
    float ni_over_nt;
    attenuation = Vec3(1.0, 1.0, 1.0);
    Vec3 refracted;
    float reflect_prob;
    float cosine;
    if (dot(r_in.direction(), rec.normal) > 0.0f) {
      outward_normal = -rec.normal;
      ni_over_nt = refraction_ratio;
      cosine = dot(r_in.direction(), rec.normal) / r_in.direction().length();
      cosine = sqrtf(1.0f - refraction_ratio*refraction_ratio*(1.f-cosine*cosine));
    }
    else {
      outward_normal = rec.normal;
      ni_over_nt = 1.0f / refraction_ratio;
      cosine = -dot(r_in.direction(), rec.normal) / r_in.direction().length();
    }
    if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted)) //ONLY REFRACT
      reflect_prob = reflectance(cosine, refraction_ratio); //REFLECTANCE CALL
    else
      reflect_prob = 1.0f;
    if (curand_uniform(local_rand_state) < reflect_prob)
      scattered = Ray(rec.p, reflected, r_in.time());
    else
      scattered = Ray(rec.p, refracted, r_in.time());
    return true;
    }

    __device__ static float reflectance(float cosine, float ref)
    {
        // Use Sclick's approximation for reflectance
        float r0 = (1.f - ref) / (1.f + ref);
        r0 *= r0;
        return r0 + (1.f - r0) * powf((1.f - cosine), 5.f);
    }

private:
    float refraction;
};

class Diffuse_Light : public Material
{
public:
    __device__ Diffuse_Light(Texture *a) : emit(a) {}
    __device__ Diffuse_Light(Color c)
    {
        emit = new Solid_Color (c);
    }

 __device__ virtual bool scatter(const Ray& r_in, hit_record& rec, Color& attenuation, Ray& scattered,  curandState *local_rand_state) const override
 {
     return false;
}

__device__ virtual Color emitted(float u, float v, const Point3& p) const override 
{
    return emit->value(u, v, p);
}

public:
    Texture *emit;
};


#endif // HITTABLE_H