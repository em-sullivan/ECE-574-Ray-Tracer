/*
 * Material Source File
 */

#include "Material.h"

bool Material::scatter(const Ray &r_in, hit_record &rec, Color &attenuation, Ray &scattered) const
{
    // Generic implementation - does nothing
    return false;
}

Lambertian::Lambertian(const Color &a)
{
    albedo = a;
}

bool Lambertian::scatter(const Ray &r_in, hit_record &rec, Color &attenuation, Ray &scattered) const
{
    auto scatter_dir = rec.normal + randomUnitVector();
    //auto scatter_dir = randomInHemisphere(rec.normal);

    // Catch degenerate scatter direction
    if (scatter_dir.nearZero())
        scatter_dir = rec.normal;

    scattered = Ray(rec.p, scatter_dir);
    attenuation = albedo;
    return true;
}

Metal::Metal(const Color &a, double in_fuzz)
{
    albedo = a;
    fuzz = in_fuzz < 1 ? in_fuzz : 1;
}

bool Metal::scatter(const Ray &r_in, hit_record &rec, Color &attenuation, Ray &scattered) const
{
    Vec3 reflected = reflect(unitVector(r_in.direction()), rec.normal);
    scattered = Ray(rec.p, reflected + fuzz * randomInUnitSphere());
    attenuation = albedo;
    return (dot(scattered.direction(), rec.normal) > 0);
}

Dielectric::Dielectric(double refraction_index)
{
    refraction = refraction_index;
}

bool Dielectric::scatter(const Ray &r_in, hit_record &rec, Color &attenuation, Ray &scattered) const
{
    attenuation = Color(1.0, 1.0, 1.0);
    double refraction_ratio = rec.front_face ? (1.0 / refraction) : refraction;

    Vec3 unit_direction = unitVector(r_in.direction());

    double cos_theta = fmin(dot(-unit_direction, rec.normal), 1.0);
    double sin_theta = sqrt(1.0 - cos_theta * cos_theta);

    // Determine if it can refract
    bool cannot_refract = refraction_ratio * sin_theta > 1.0;
    Vec3 direction;

    if (cannot_refract || reflectance(cos_theta, refraction_ratio) > random_double())
        direction = reflect(unit_direction, rec.normal);
    else
        // Reflect if it can't refract
        direction = refract(unit_direction, rec.normal, refraction_ratio);

    scattered = Ray(rec.p, direction);

    return true;
}


double Dielectric::reflectance(double cosine, double ref)
{
    // Use Sclick's approximation for reflectance
    auto r0 = (1 - ref) / (1 + ref);
    r0 *= r0;
    return r0 + (1 - r0) * pow((1 - cosine), 5);
}
