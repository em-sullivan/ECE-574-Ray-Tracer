/*
 * Main program: Create a random world of Balls
 */

#include <fstream>
#include <iostream>
#include <string>
#include "shader_consts.h"
#include "Color.h"
#include "Hittable_List.h"
#include "Sphere.h"
#include "Camera.h"
#include "Material.h"
#include "Moving_Sphere.h"

Hittable_List random_balls()
{
    Hittable_List world;

    // Gray ground
    auto ground_material = make_shared<Lambertian>(Color(0.5, 0.5, 0.5));
    world.add(make_shared<Sphere>(Point3(0, -1000, 0), 1000, ground_material));

    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {

            // "Randomly" choose material and point
            auto choose_mat = random_float();
            Point3 center(a + 0.9 * random_float(), 0.2, b + 0.9 * random_float());

            if ((center - Point3(4, 0.2, 0)).length() > 0.9) {
                shared_ptr<Material> sphere_material;

                if (choose_mat < 0.8) {
                    // Diffuse ball
                    auto albedo = Color::random() * Color::random();
                    sphere_material = make_shared<Lambertian>(albedo);
                    world.add(make_shared<Sphere>(center, 0.2, sphere_material));
                } else if (choose_mat < 0.95) {
                    // Metal ball
                    auto albedo = Color::random(0.5, 1);
                    auto fuzz = random_float(0, 0.5);
                    sphere_material = make_shared<Metal>(albedo, fuzz);
                    world.add(make_shared<Sphere>(center, 0.2, sphere_material));
                } else {
                    // Glass ball
                    sphere_material = make_shared<Dielectric>(1.5);
                    world.add(make_shared<Sphere>(center, 0.2, sphere_material));
                }
            }

        }
    }

    // Add some big chonking balls
    auto ball1 =make_shared<Dielectric>(1.5);
    world.add(make_shared<Sphere>(Point3(0, 1, 0), 1.0, ball1));

    auto ball2 = make_shared<Lambertian>(Color(0.4, 0.2, 0.1));
    world.add(make_shared<Sphere>(Point3(-4, 1, 0), 1.0, ball2));

    auto ball3 = make_shared<Metal>(Color(0.7, 0.6, 0.5), 0.0);
    world.add(make_shared<Sphere>(Point3(4, 1, 0), 1.0, ball3));

    return world;
}

Hittable_List three_balls()
{
    // Make a simple world of three balls
    // For Eric's poor slow computer :(
    Hittable_List world;

    auto ground = make_shared<Lambertian>(Color(0.8, 0.8, 0.0));
    auto center = make_shared<Lambertian>(Color(0.1, 0.2, 0.5));
    auto tiny = make_shared<Lambertian>(Color(1, 0, 1));
    auto left = make_shared<Dielectric>(1.25);
    auto right = make_shared<Metal>(Color(0.8, 0.6, 0.2), 0.0);

    // Green ball ground
    // Blue ball sandwiched between glass and metal balls
    world.add(make_shared<Sphere>(Point3(0.0, -100.5, -1.0), 100.0, ground));
    world.add(make_shared<Sphere>(Point3(0.0, 0.0, -1.0), 0.5, center));
    world.add(make_shared<Sphere>(Point3(-1.0, 0.0, -1.0), 0.5, left));
    world.add(make_shared<Sphere>(Point3(1.0, 0.0, -1.0), 0.5, right));

    // Tiny lil guy moving around
    world.add(make_shared<Moving_Sphere>(Point3(0.1, -0.5, 1.0), Point3(0.0, -0.5, 1.0),
        0.0, 1.0, 0.1, tiny));

    return world;
}

Color ray_color(const Ray &r, const Hittable &world, int depth)
{
    hit_record rec;

    // If the ray bounce limit is exceeded, no more light is gathered
    if (depth <= 0)
        return Color(0,0,0);

    if (world.hit(r, 0.001f, INF, rec)) {
        Ray scattered;
        Color attenuation;
        if (rec.mat_ptr->scatter(r, rec, attenuation, scattered))
            return attenuation * ray_color(scattered, world, depth-1);

        return Color(0, 0, 0);
    }

    // This prints the blueish-white sky
    Vec3 unit_direction = unitVector(r.direction());
    float t = 0.5 * (unit_direction.y() + 1.0);
    return (1.0 - t) * Color(1.0, 1.0, 1.0) + t * Color(0.5, 0.7, 1.0);
}


int main(int argc, char **argv)
{
    // Image
    const float aspect_ratio = 16.0f / 9.0f;
    int image_width = 400;
    int image_height = static_cast<int>(image_width / aspect_ratio);
    int samples_per_pixel = 500;
    int max_depth = 50;

    // World
    // Generate three balls
    auto world = three_balls();
        
    // This worlds camera
    auto lookfrom = Point3(0, 0, 5);
    auto lookat = Point3(0, 0, -1);
    auto vup = Vec3(0, 1, 0);
    auto fov = 20;
    float dist_to_focus = (lookfrom-lookat).length();
    float aperture = 0.1;

    /*
    // Generate random balls - takes a while!
    auto world = random_balls();

    // Many balls camera
    auto lookfrom = Point3(13, 2, 3);
    auto lookat = Point3(0, 0, 0);
    auto vup = Vec3(0, 1, 0);
    auto fov = 20;
    auto dist_to_focus = 10.0;
    auto aperture = .1;
    */

    // Camera
    Camera cam(lookfrom, lookat, vup, fov, aspect_ratio, aperture, dist_to_focus, 0.0, 1.0);

    // Output File
    std::fstream file;
    file.open("out.ppm", std::ios::out);
    std::streambuf *ppm_out = file.rdbuf();

    // Redirect Cout
    std::cout.rdbuf(ppm_out);
    
    // Render
    std::cout << "P3\n" << image_width << " " << image_height << "\n255\n";

    for (int j = image_height - 1; j >= 0; j--) {
        
        // Progress indicator
        std::cerr << "\rScanline remaining: " << j << ' ' << std::flush;

        for (int i = 0; i < image_width; i++) {
            
            Color pixel_color(0, 0, 0);
            for(int s = 0; s < samples_per_pixel; s++) {
                float u = (i + random_float()) / (image_width - 1);
                float v = (j + random_float()) / (image_height - 1);
                Ray r = cam.get_ray(u, v);
                pixel_color += ray_color(r, world, max_depth);
            }
            writeColor(std::cout, pixel_color, samples_per_pixel);
        }
        
    }

    std::cerr << "\nDone" << std::endl;
    file.close();
    return 0;
}