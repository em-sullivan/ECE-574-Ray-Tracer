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
    __device__ virtual bool scatter(const Ray &r_in, hit_record &rec, Color &attenuation, Ray &scattered) const;
    __device__ virtual Color emitted(float u, float v, const Point3& p) const; 
};

class Lambertian : public Material
{
public:
    // Constructor
    __device__  Lambertian(const Color &a) : albedo(make_shared<Solid_Color>(a)){}
    __device__  Lambertian(shared_ptr<Texture> a);

    __device__  virtual bool scatter(const Ray &r_in, hit_record &rec, Color &attenuation, Ray &scattered) const override;

private:
    //Color albedo;
    shared_ptr<Texture> albedo;
};

class Metal : public Material
{
public:
    // Constructor
    __device__  Metal(const Color &a, float in_fuzz);

    __device__  virtual bool scatter(const Ray &r_in, hit_record &rec, Color &attenuation, Ray &scattered) const override;

private:
    Color albedo;
    float fuzz;
};

class Dielectric : public Material
{
public:
    __device__  Dielectric(float refraction_index);
    __device__  virtual bool scatter(const Ray &r_in, hit_record &rec, Color &attenuation, Ray &scattered) const override;
    __device__  static float reflectance(float cosine, float ref);

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

class Diffuse_Light : public Material
{
public:
    __device__  Diffuse_Light(shared_ptr<Texture> a) : emit(a) {}
    __device__  Diffuse_Light(Color c) : emit(make_shared<Solid_Color>(c)) {}

    __device__  virtual bool scatter(const Ray& r_in, hit_record& rec, Color& attenuation, Ray& scattered) const override;
    __device__ virtual Color emitted(float u, float v, const Point3& p) const override;

private:
    shared_ptr<Texture> emit;
};

class Isotropic : public Material
{
public:
    __device__  Isotropic(Color c) : albedo(make_shared<Solid_Color>(c)) {}
    __device__  Isotropic(shared_ptr<Texture> a) : albedo(a) {}

    __device__  virtual bool scatter(const Ray& r_in, hit_record& rec, Color& attenuation, Ray& scattered) const override;

private:
    shared_ptr<Texture> albedo;
};



#endif // MATERIAL_H