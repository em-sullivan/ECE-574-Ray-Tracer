/*
 * Material Class
 */

#ifndef MATERIAL_H
#define MATERIAL_H

#include "Vec3.h"
#include "Ray.h"
#include "Hittable.h"
#include "Texture.h"

struct hit_record;

class Material
{
public:
    virtual bool scatter(const Ray &r_in, hit_record &rec, Color &attenuation, Ray &scattered) const;
};

class Lambertian : public Material
{
public:
    // Constructor
    Lambertian(const Color &a) : albedo(make_shared<Solid_Color>(a)){}
    Lambertian(shared_ptr<Texture> a);

    virtual bool scatter(const Ray &r_in, hit_record &rec, Color &attenuation, Ray &scattered) const override;

private:
    //Color albedo;
    shared_ptr<Texture> albedo;
};

class Metal : public Material
{
public:
    // Constructor
    Metal(const Color &a, float in_fuzz);

    virtual bool scatter(const Ray &r_in, hit_record &rec, Color &attenuation, Ray &scattered) const override;

private:
    Color albedo;
    float fuzz;
};

class Dielectric : public Material
{
public:
    Dielectric(float refraction_index);
    virtual bool scatter(const Ray &r_in, hit_record &rec, Color &attenuation, Ray &scattered) const override;
    static float reflectance(float cosine, float ref);

private:
    float refraction;
    /*
    static float reflectance(float cosine, float ref)
    {
        // Use Sclick's approximation for reflectance
        auto r0 = (1 - ref) / (1 + ref);
        r0 *= r0;
        return r0 + (1 - r0) * pow((1 - cosine), 5);
    }*/
};

#endif // MATERIAL_H