/*
 * Material Source File
 */

#include "Material.h"

__device__ bool Material::scatter(const Ray &r_in, hit_record &rec, Color &attenuation, Ray &scattered) const
{
    // Generic implementation - does nothing
    return false;
}

__device__ Color Material::emitted(float u, float v, const Point3& p) const
{
    return Color(0,0,0);
}

//Lambertian::Lambertian(const Color &a)
//{
//    albedo = make_shared<Solid_Color>(a);
//}

__device__ Lambertian::Lambertian(shared_ptr<Texture> a)
{
    albedo = a;
}

__device__  bool Lambertian::scatter(const Ray &r_in, hit_record &rec, Color &attenuation, Ray &scattered) const
{
    auto scatter_dir = rec.normal + randomUnitVector();
    //auto scatter_dir = randomInHemisphere(rec.normal);

    // Catch degenerate scatter direction
    if (scatter_dir.nearZero())
        scatter_dir = rec.normal;

    scattered = Ray(rec.p, scatter_dir, r_in.time());
    attenuation = albedo->value(rec.u, rec.v, rec.p);
    return true;
}

__device__ Metal::Metal(const Color &a, float in_fuzz)
{
    albedo = a;
    fuzz = in_fuzz < 1 ? in_fuzz : 1;
}

__device__ bool Metal::scatter(const Ray &r_in, hit_record &rec, Color &attenuation, Ray &scattered) const
{
    Vec3 reflected = reflect(unitVector(r_in.direction()), rec.normal);
    scattered = Ray(rec.p, reflected + fuzz * randomInUnitSphere(), r_in.time());
    attenuation = albedo;
    return (dot(scattered.direction(), rec.normal) > 0);
}

__device__ Dielectric::Dielectric(float refraction_index)
{
    refraction = refraction_index;
}

__device__ bool Dielectric::scatter(const Ray &r_in, hit_record &rec, Color &attenuation, Ray &scattered) const
{
    attenuation = Color(1.0, 1.0, 1.0);
    float refraction_ratio = rec.front_face ? (1.0 / refraction) : refraction;

    Vec3 unit_direction = unitVector(r_in.direction());

    float cos_theta = fmin(dot(-unit_direction, rec.normal), 1.0);
    float sin_theta = sqrt(1.0 - cos_theta * cos_theta);

    // Determine if it can refract
    bool cannot_refract = refraction_ratio * sin_theta > 1.0;
    Vec3 direction;

    if (cannot_refract || reflectance(cos_theta, refraction_ratio) > random_float())
        direction = reflect(unit_direction, rec.normal);
    else
        // Reflect if it can't refract
        direction = refract(unit_direction, rec.normal, refraction_ratio);

    scattered = Ray(rec.p, direction, r_in.time());

    return true;
}


__device__ float Dielectric::reflectance(float cosine, float ref)
{
    // Use Sclick's approximation for reflectance
    float r0 = (1 - ref) / (1 + ref);
    r0 *= r0;
    return r0 + (1 - r0) * pow((1 - cosine), 5);
}

__device__ bool Diffuse_Light::scatter(const Ray& r_in, hit_record& rec, Color& attenuation, Ray& scattered) const
{
    return false;
}

__device__ Color Diffuse_Light::emitted(float u, float v, const Point3& p) const
{
    return emit->value(u, v, p);
}

__device__ bool Isotropic::scatter(const Ray& r_in, hit_record& rec, Color& attenuation, Ray& scattered) const
{
    scattered = Ray(rec.p, randomInUnitSphere(), r_in.time());
    attenuation = albedo->value(rec.u, rec.v, rec.p);
    return true;
}