/*
 * Main program: Create a random world of Balls
 */

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include "shader_consts.h"
#include "Color.h"
#include "Hittable_List.h"
#include "Sphere.h"
#include "Camera.h"
#include "Material.h"
#include "Moving_Sphere.h"
#include "Perlin.h"
#include "Aarect.h"
#include "Box.h"
#include "Translate.h"
#include "Constant_Medium.h"
#include "Bvh_Node.h"

Hittable_List random_balls()
{
    Hittable_List world;

    // Gray ground
    auto ground_material = new Lambertian(Color(0.5, 0.5, 0.5));
    world.add(new Sphere(Point3(0, -1000, 0), 1000, ground_material));

    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {

            // "Randomly" choose material and point
            auto choose_mat = random_float();
            Point3 center(a + 0.9 * random_float(), 0.2, b + 0.9 * random_float());

            if ((center - Point3(4, 0.2, 0)).length() > 0.9) {
                Material *sphere_material;

                if (choose_mat < 0.8) {
                    // Diffuse ball
                    auto albedo = Color::random() * Color::random();
                    sphere_material = new Lambertian(albedo);
                    world.add(new Sphere(center, 0.2, sphere_material));
                } else if (choose_mat < 0.95) {
                    // Metal ball
                    auto albedo = Color::random(0.5, 1);
                    auto fuzz = random_float(0, 0.5);
                    sphere_material = new Metal(albedo, fuzz);
                    world.add(new Sphere(center, 0.2, sphere_material));
                } else {
                    // Glass ball
                    sphere_material = new Dielectric(1.5);
                    world.add(new Sphere(center, 0.2, sphere_material));
                }
            }

        }
    }

    // Add some big chonking balls
    auto ball1 = new Dielectric(1.5);
    world.add(new Sphere(Point3(0, 1, 0), 1.0, ball1));

    auto ball2 = new Lambertian(Color(0.4, 0.2, 0.1));
    world.add(new Sphere(Point3(-4, 1, 0), 1.0, ball2));

    auto ball3 = new Metal(Color(0.7, 0.6, 0.5), 0.0);
    world.add(new Sphere(Point3(4, 1, 0), 1.0, ball3));

    return world;
}

Hittable_List three_balls()
{
    // Make a simple world of three balls
    // For Eric's poor slow computer :(
    Hittable_List world;

    auto ground = new Checkered(Color(0.8, 0.8, 0.0), Color(0, 0, 0));
    auto center = new Lambertian(Color(0.1, 0.2, 0.5));
    auto tiny = new Lambertian(Color(1, 0, 1));
    auto left = new Dielectric(1.25);
    auto right = new Metal(Color(0.8, 0.6, 0.2), 0.0);

    // Green ball ground
    // Blue ball sandwiched between glass and metal balls
    world.add(new Sphere(Point3(0.0, -100.5, -1.0), 100.0, new Lambertian(ground)));
    world.add(new Sphere(Point3(0.0, 0.0, -1.0), 0.5, center));
    world.add(new Sphere(Point3(-1.0, 0.0, -1.0), 0.5, left));
    world.add(new Sphere(Point3(1.0, 0.0, -1.0), 0.5, right));

    // Tiny lil guy moving around
    world.add(new Moving_Sphere(Point3(0.1, -0.5, 1.0), Point3(0.0, -0.5, 1.0),
        0.0, 1.0, 0.1, tiny));

    return world;
}

Hittable_List two_bit_balls()
{
    Hittable_List objects;

    auto checker = new Checkered(Color(0.9, 0.9, 0.9), Color(1, 0, 1));

    objects.add(new Sphere(Point3(0, -10, 0), 10, new Lambertian(checker)));
    objects.add(new Sphere(Point3(0, 10, 0), 10, new Lambertian(checker)));

    return objects;
}

Hittable_List two_fuzzy_balls()
{
    Hittable_List world;

    auto fuzz = new Noise_Text(4);
    world.add(new Sphere(Point3(0, -1000, 0), 1000, new Lambertian(fuzz)));
    world.add(new Sphere(Point3(0, 2, 0), 2, new Lambertian(fuzz)));
    
    return world;
}

Hittable_List simple_light() 
{
    Hittable_List objects;

    auto pertext = new Noise_Text(4);
    objects.add(new Sphere(Point3(0, -1000, 0), 1000, new Lambertian(pertext)));
    objects.add(new Sphere(Point3(0, 2, 0), 2, new Lambertian(pertext)));

    auto difflight = new Diffuse_Light(Color(4, 4, 4));
    objects.add(new Sphere(Point3(0, 7, 0), 2, difflight));
    objects.add(new XY_Rect(3, 5, 1, 3, -2, difflight));

    return objects;
}

Hittable_List cornell_box()
{
    Hittable_List objects;

    auto red   = new Lambertian(Color(.65, .05, .05));
    auto white = new Lambertian(Color(.73, .73, .73));
    auto green = new Lambertian(Color(.12, .45, .15));
    auto light = new Diffuse_Light(Color(15, 15, 15));


    objects.add(new YZ_Rect(0, 555, 0, 555, 555, green));
    objects.add(new YZ_Rect(0, 555, 0, 555, 0, red));
    objects.add(new XZ_Rect(213, 343, 227, 332, 554, light));
    objects.add(new XZ_Rect(0, 555, 0, 555, 555, white));
    objects.add(new XZ_Rect(0, 555, 0, 555, 0, white));
    objects.add(new XY_Rect(0, 555, 0, 555, 555, white));

    Hittable *box1 = new Box(Point3(0, 0, 0), Point3(165, 330, 165), white);
    box1 = new Rotate_Y(box1, 15);
    box1 = new Translate(box1, Vec3(265,0,295));
    objects.add(box1);

    Hittable *box2 = new Box(Point3(0,0,0), Point3(165,165,165), white);
    box2 = new Rotate_Y(box2, -18);
    box2 = new Translate(box2, Vec3(130,0,65));
    objects.add(box2);

    return objects;
}

Hittable_List cornell_smoke() {
    Hittable_List objects;

    auto red   = new Lambertian(Color(.65, .05, .05));
    auto white = new Lambertian(Color(.73, .73, .73));
    auto green = new Lambertian(Color(.12, .45, .15));
    auto light = new Diffuse_Light(Color(7, 7, 7));

    objects.add(new YZ_Rect(0, 555, 0, 555, 555, green));
    objects.add(new YZ_Rect(0, 555, 0, 555, 0, red));
    objects.add(new XZ_Rect(113, 443, 127, 432, 554, light));
    objects.add(new XZ_Rect(0, 555, 0, 555, 555, white));
    objects.add(new XZ_Rect(0, 555, 0, 555, 0, white));
    objects.add(new XY_Rect(0, 555, 0, 555, 555, white));

    Hittable *box1 = new Box(Point3(0,0,0), Point3(165,330,165), white);
    box1 = new Rotate_Y(box1, 15);
    box1 = new Translate(box1, Vec3(265,0,295));

    Hittable *box2 = new Box(Point3(0,0,0), Point3(165,165,165), white);
    box2 = new Rotate_Y(box2, -18);
    box2 = new Translate(box2, Vec3(130,0,65));

    objects.add(new Constant_Medium(box1, 0.01, Color(0,0,0)));
    objects.add(new Constant_Medium(box2, 0.01, Color(1,1,1)));

    return objects;
}

Hittable_List final_scene()
{
    Hittable_List boxes1;
    auto ground = new Lambertian(Color(0.48, 0.83, 0.53));


    const int boxes_per_side = 20;
    for (int i = 0; i < boxes_per_side; i++) {
        for (int j = 0; j < boxes_per_side; j++) {
            auto w = 100.0;
            auto x0 = -1000.0 + i*w;
            auto z0 = -1000.0 + j*w;
            auto y0 = 0.0;
            auto x1 = x0 + w;
            auto y1 = random_float(1,101);
            auto z1 = z0 + w;

            boxes1.add(new Box(Point3(x0,y0,z0), Point3(x1,y1,z1), ground));
        }
    }

    Hittable_List objects;

    objects.add(new Bvh_Node(boxes1, 0, 1));

    auto light = new Diffuse_Light(Color(7, 7, 7));
    objects.add(new XZ_Rect(123, 423, 147, 412, 554, light));

    auto center1 = Point3(400, 400, 200);
    auto center2 = center1 + Vec3(30,0,0);
    auto moving_sphere_material = new Lambertian(Color(0.7, 0.3, 0.1));
    objects.add(new Moving_Sphere(center1, center2, 0, 1, 50, moving_sphere_material));

    objects.add(new Sphere(Point3(260, 150, 45), 50, new Dielectric(1.5)));
    objects.add(new Sphere(Point3(0, 150, 145), 50, new Metal(Color(0.8, 0.8, 0.9), 1.0)));

    auto boundary = new Sphere(Point3(360,150,145), 70, new Dielectric(1.5));
    objects.add(boundary);
    objects.add(new Constant_Medium(boundary, 0.2, Color(0.2, 0.4, 0.9)));
    boundary = new Sphere(Point3(0, 0, 0), 5000, new Dielectric(1.5));
    objects.add(new Constant_Medium(boundary, .0001, Color(1,1,1)));

    auto emat = new Lambertian(new Image_Text("textures/earthmap.jpg"));
    objects.add(new Sphere(Point3(400,200,400), 100, emat));
    auto pertext = new Noise_Text(0.1);
    objects.add(new Sphere(Point3(220,280,300), 80, new Lambertian(pertext)));

    Hittable_List boxes2;
    auto white = new Lambertian(Color(.73, .73, .73));
    int ns = 1000;
    for (int j = 0; j < ns; j++) {
        boxes2.add(new Sphere(Point3::random(0,165), 10, white));
    }

    objects.add(new Translate(new Rotate_Y(new Bvh_Node(boxes2, 0.0, 1.0), 15),Vec3(-100,270,395)));

    return objects;
}

Color ray_color(const Ray &r, const Color& background, const Hittable &world, int depth)
{
    hit_record rec;

    // If the ray bounce limit is exceeded, no more light is gathered
    if (depth <= 0)
        return Color(0, 0, 0);

    // If the ray didn't hit anything, return the background
    if (!world.hit(r, 0.001f, INF, rec)) {
        return background;
    }
    
    Ray scattered;
    Color attenuation;
    Color emitted = rec.mat_ptr->emitted(rec.u,rec.v,rec.p);

    if (rec.mat_ptr->scatter(r, rec, attenuation, scattered)) {
        return emitted + attenuation * ray_color(scattered, background, world, depth-1);
    }
        return emitted;
}
/*
    // This prints the blueish-white sky
    Vec3 unit_direction = unitVector(r.direction());
    float t = 0.5 * (unit_direction.y() + 1.0);
    return (1.0 - t) * Color(1.0, 1.0, 1.0) + t * Color(0.5, 0.7, 1.0);
}
*/

int main(int argc, char **argv)
{
    int image;
    if (argc < 2) {
        image = 0;
    } else {
        std::stringstream str_to_int(argv[1]);
        str_to_int >> image;
    }
    
    // Image
    float aspect_ratio = 16.0f / 9.0f;
    int image_width = 400;
    int samples_per_pixel = 50;
    int max_depth = 50;
    int image_height;

    // World
    Hittable_List world;
    Point3 lookfrom;
    Point3 lookat;
    Point3 vup;
    Color background;
    float fov;
    float aperture;
    float dist_to_focus;


    switch(image) {
        case 1:
            // Generate three balls
            world = three_balls();
        
            // This worlds camera
            lookfrom = Point3(0, 0, 5);
            lookat = Point3(0, 0, -1);
            vup = Vec3(0, 1, 0);
            background = Color(0.70, 0.80, 1.00);
            fov = 20;
            dist_to_focus = (lookfrom-lookat).length();
            aperture = 0.1;
            break;

        case 2:
            // Generate random balls - takes a while!
            world = random_balls();

            lookfrom = Point3(13, 2, 3);
            lookat = Point3(0, 0, 0);
            vup = Vec3(0, 1, 0);
            background = Color(0.70, 0.80, 1.00);
            fov = 20;
            dist_to_focus = 10.0;
            aperture = .1;
            break;

        case 3:
            // Generates two fuzzy balls
            world = two_fuzzy_balls();
            lookfrom = Point3(13, 2, 3);
            lookat = Point3(0, 0, 0);
            vup = Vec3(0, 1, 0);
            background = Color(0.70, 0.80, 1.00);
            fov = 20;
            dist_to_focus = 10.0;
            aperture = 0;
            break;
            

        case 4:
            world = two_bit_balls();
            lookfrom = Point3(13, 2, 3);
            lookat = Point3(0, 0, 0);
            vup = Vec3(0, 1, 0);
            background = Color(0.70, 0.80, 1.00);
            fov = 20;
            dist_to_focus = 10.0;
            aperture = 0;
            break;


        case 5:
            world = simple_light();
            lookfrom = Point3(26, 3, 6);
            lookat = Point3(0, 2, 0);
            vup = Vec3(0, 1, 0);
            background = Color(0, 0, 0);
            fov = 20;
            dist_to_focus = 10.0;
            aperture = 0;
            break;

        case 6:
            world = cornell_box();
            aspect_ratio = 1.0;
            image_width = 300;
            lookfrom = Point3(278, 278, -800);
            lookat = Point3(278, 278, 0);
            vup = Vec3(0, 1, 0);
            fov = 40;
            dist_to_focus = 10.0;
            aperture = 0;
            break;
        
        case 7:
            world = cornell_smoke();
            aspect_ratio = 1.0;
            image_width = 300;
            lookfrom = Point3(278, 278, -800);
            lookat = Point3(278, 278, 0);
            vup = Vec3(0, 1, 0);
            fov = 40.0;
            dist_to_focus = 10.0;
            aperture = 0;
            break;

        default:
        case 8:
            world = final_scene();
            aspect_ratio = 1.0;
            image_width = 400;
            background = Color(0,0,0);
            lookfrom = Point3(478, 278, -600);
            lookat = Point3(278, 278, 0);
            vup = Vec3(0, 1, 0);
            fov = 40.0;
            dist_to_focus = 10.0;
            aperture = 0;
            break;
    }

    // Camera
    Camera cam(lookfrom, lookat, vup, fov, aspect_ratio, aperture, dist_to_focus, 0.0, 1.0);
    image_height = static_cast<int>(image_width / aspect_ratio);

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
                pixel_color += ray_color(r, background, world, max_depth);
            }
            writeColor(std::cout, pixel_color, samples_per_pixel);
        }
        
    }

    std::cerr << "\nDone" << std::endl;
    file.close();
    return 0;
}