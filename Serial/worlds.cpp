/*
 * This file contains a collection
 * of worlds/scenese in the ray tracer. Each world has
 * a world function and a camera function
 */

#include "worlds.h"

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

Camera random_balls_cam(float aspect_ratio)
{
    Point3 lookfrom = Point3(13, 2, 3);
    Point3 lookat = Point3(0, 0, 0);
    Point3 vup = Vec3(0, 1, 0);
    Color background = Color(0.70, 0.80, 1.00);
    float fov = 20;
    float dist_to_focus = 10.0;
    float aperture = .1;

    return Camera(lookfrom, lookat, vup, fov, aspect_ratio, aperture, dist_to_focus, 0.0, 1.0);
}

Hittable_List three_balls()
{
    // Make a simple world of three balls
    // For Eric's poor slow computer :(
    Hittable_List world;

    auto ground = make_shared<Checkered>(Color(0.8, 0.8, 0.0), Color(0, 0, 0));
    auto center = make_shared<Lambertian>(Color(0.1, 0.2, 0.5));
    auto tiny = make_shared<Lambertian>(Color(1, 0, 1));
    auto left = make_shared<Dielectric>(1.25);
    auto right = make_shared<Metal>(Color(0.8, 0.6, 0.2), 0.0);

    // Green ball ground
    // Blue ball sandwiched between glass and metal balls
    world.add(make_shared<Sphere>(Point3(0.0, -100.5, -1.0), 100.0, make_shared<Lambertian>(ground)));
    world.add(make_shared<Sphere>(Point3(0.0, 0.0, -1.0), 0.5, center));
    world.add(make_shared<Sphere>(Point3(-1.0, 0.0, -1.0), 0.5, left));
    world.add(make_shared<Sphere>(Point3(1.0, 0.0, -1.0), 0.5, right));

    // Tiny lil guy moving around
    world.add(make_shared<Moving_Sphere>(Point3(0.1, -0.5, 1.0), Point3(0.0, -0.5, 1.0),
        0.0, 1.0, 0.1, tiny));

    return world;
}

Camera three_balls_cam(float aspect_ratio)
{
    Point3 lookfrom = Point3(0, 0, 5);
    Point3 lookat = Point3(0, 0, -1);
    Point3 vup = Vec3(0, 1, 0);
    Color background = Color(0.70, 0.80, 1.00);
    float fov = 20;
    float dist_to_focus = (lookfrom-lookat).length();
    float aperture = 0;

    return Camera(lookfrom, lookat, vup, fov, aspect_ratio, aperture, dist_to_focus, 0.0, 1.0);
}

Hittable_List two_bit_balls()
{
    Hittable_List objects;

    auto checker = make_shared<Checkered>(Color(0.9, 0.9, 0.9), Color(1, 0, 1));

    objects.add(make_shared<Sphere>(Point3(0, -10, 0), 10, make_shared<Lambertian>(checker)));
    objects.add(make_shared<Sphere>(Point3(0, 10, 0), 10, make_shared<Lambertian>(checker)));

    return objects;
}

Camera two_bit_balls_cam(float aspect_ratio)
{
    Point3 lookfrom = Point3(13, 2, 3);
    Point3 lookat = Point3(0, 0, 0);
    Point3 vup = Vec3(0, 1, 0);
    Color background = Color(0.70, 0.80, 1.00);
    float fov = 20;
    float dist_to_focus = 10.0;
    float aperture = 0;

    return Camera(lookfrom, lookat, vup, fov, aspect_ratio, aperture, dist_to_focus, 0.0, 1.0);
}

Hittable_List two_fuzzy_balls()
{
    Hittable_List world;

    auto fuzz = make_shared<Noise_Text>(4);
    world.add(make_shared<Sphere>(Point3(0, -1000, 0), 1000, make_shared<Lambertian>(fuzz)));
    world.add(make_shared<Sphere>(Point3(0, 2, 0), 2, make_shared<Lambertian>(fuzz)));
    
    return world;
}

Camera two_fuzzy_balls_cam(float aspect_ratio)
{
    auto lookfrom = Point3(13, 2, 3);
    auto lookat = Point3(0, 0, 0);
    auto vup = Vec3(0, 1, 0);
    auto background = Color(0.70, 0.80, 1.00);
    float fov = 20;
    float dist_to_focus = 10.0;
    float aperture = 0;

    return Camera(lookfrom, lookat, vup, fov, aspect_ratio, aperture, dist_to_focus, 0.0, 1.0);
}


Hittable_List earf()
{
    auto earf_texture = make_shared<Image_Text>("textures/earthmap.jpg");
    auto earf_surface = make_shared<Lambertian>(earf_texture);
    auto globe = make_shared<Sphere>(Point3(0, 0, 0), 2, earf_surface);

    return Hittable_List(globe);
}

Camera earf_cam(float aspect_ratio)
{
    auto lookfrom = Point3(13, 2, 3);
    auto lookat = Point3(0, 0, 0);
    auto vup = Vec3(0, 1, 0);
    auto background = Color(0.70, 0.80, 1.00);
    float fov = 20;
    float dist_to_focus = 10.0;
    float aperture = 0;

    return Camera(lookfrom, lookat, vup, fov, aspect_ratio, aperture, dist_to_focus, 0.0, 1.0);
}

Hittable_List simple_light() 
{
    Hittable_List objects;

    auto pertext = make_shared<Noise_Text>(4);
    objects.add(make_shared<Sphere>(Point3(0, -1000, 0), 1000, make_shared<Lambertian>(pertext)));
    objects.add(make_shared<Sphere>(Point3(0, 2, 0), 2, make_shared<Lambertian>(pertext)));

    auto difflight = make_shared<Diffuse_Light>(Color(4, 4, 4));
    objects.add(make_shared<Sphere>(Point3(0, 7, 0), 2, difflight));
    objects.add(make_shared<XY_Rect>(3, 5, 1, 3, -2, difflight));

    return objects;
}

Camera simple_light_cam(float aspect_ratio)
{
    auto lookfrom = Point3(26, 3, 6);
    auto lookat = Point3(0, 2, 0);
    auto vup = Vec3(0, 1, 0);
    auto background = Color(0, 0, 0);
    float fov = 20;
    float dist_to_focus = 10.0;
    float aperture = 0;

    return Camera(lookfrom, lookat, vup, fov, aspect_ratio, aperture, dist_to_focus, 0.0, 1.0);
}

Hittable_List cornell_box()
{
    Hittable_List objects;

    auto red   = make_shared<Lambertian>(Color(.65, .05, .05));
    auto white = make_shared<Lambertian>(Color(.73, .73, .73));
    auto green = make_shared<Lambertian>(Color(.12, .45, .15));
    auto light = make_shared<Diffuse_Light>(Color(15, 15, 15));


    objects.add(make_shared<YZ_Rect>(0, 555, 0, 555, 555, green));
    objects.add(make_shared<YZ_Rect>(0, 555, 0, 555, 0, red));
    objects.add(make_shared<XZ_Rect>(213, 343, 227, 332, 554, light));
    objects.add(make_shared<XZ_Rect>(0, 555, 0, 555, 555, white));
    objects.add(make_shared<XZ_Rect>(0, 555, 0, 555, 0, white));
    objects.add(make_shared<XY_Rect>(0, 555, 0, 555, 555, white));

    shared_ptr<Hittable> box1 = make_shared<Box>(Point3(0, 0, 0), Point3(165, 330, 165), white);
    box1 = make_shared<Rotate_Y>(box1, 15);
    box1 = make_shared<Translate>(box1, Vec3(265,0,295));
    objects.add(box1);

    shared_ptr<Hittable> box2 = make_shared<Box>(Point3(0,0,0), Point3(165,165,165), white);
    box2 = make_shared<Rotate_Y>(box2, -18);
    box2 = make_shared<Translate>(box2, Vec3(130,0,65));
    objects.add(box2);

    return objects;
}

Camera cornell_box_cam(float aspect_ratio)
{
    auto lookfrom = Point3(278, 278, -800);
    auto lookat = Point3(278, 278, 0);
    auto vup = Vec3(0, 1, 0);
    float fov = 40;
    float dist_to_focus = 10.0;
    float aperture = 0;

    return Camera(lookfrom, lookat, vup, fov, aspect_ratio, aperture, dist_to_focus, 0.0, 1.0);
}

Hittable_List cornell_smoke() {
    Hittable_List objects;

    auto red   = make_shared<Lambertian>(Color(.65, .05, .05));
    auto white = make_shared<Lambertian>(Color(.73, .73, .73));
    auto green = make_shared<Lambertian>(Color(.12, .45, .15));
    auto light = make_shared<Diffuse_Light>(Color(7, 7, 7));

    objects.add(make_shared<YZ_Rect>(0, 555, 0, 555, 555, green));
    objects.add(make_shared<YZ_Rect>(0, 555, 0, 555, 0, red));
    objects.add(make_shared<XZ_Rect>(113, 443, 127, 432, 554, light));
    objects.add(make_shared<XZ_Rect>(0, 555, 0, 555, 555, white));
    objects.add(make_shared<XZ_Rect>(0, 555, 0, 555, 0, white));
    objects.add(make_shared<XY_Rect>(0, 555, 0, 555, 555, white));

    shared_ptr<Hittable> box1 = make_shared<Box>(Point3(0,0,0), Point3(165,330,165), white);
    box1 = make_shared<Rotate_Y>(box1, 15);
    box1 = make_shared<Translate>(box1, Vec3(265,0,295));

    shared_ptr<Hittable> box2 = make_shared<Box>(Point3(0,0,0), Point3(165,165,165), white);
    box2 = make_shared<Rotate_Y>(box2, -18);
    box2 = make_shared<Translate>(box2, Vec3(130,0,65));

    objects.add(make_shared<Constant_Medium>(box1, 0.01, Color(0,0,0)));
    objects.add(make_shared<Constant_Medium>(box2, 0.01, Color(1,1,1)));

    return objects;
}

Camera cornell_smoke_cam(float aspect_ratio)
{
    auto lookfrom = Point3(278, 278, -800);
    auto lookat = Point3(278, 278, 0);
    auto vup = Vec3(0, 1, 0);
    float fov = 40.0;
    float dist_to_focus = 10.0;
    float aperture = 0;

    return Camera(lookfrom, lookat, vup, fov, aspect_ratio, aperture, dist_to_focus, 0.0, 1.0);
}

Hittable_List final_scene()
{
    Hittable_List boxes1;
    auto ground = make_shared<Lambertian>(Color(0.48, 0.83, 0.53));


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

            boxes1.add(make_shared<Box>(Point3(x0,y0,z0), Point3(x1,y1,z1), ground));
        }
    }

    Hittable_List objects;

    objects.add(make_shared<Bvh_Node>(boxes1, 0, 1));

    auto light = make_shared<Diffuse_Light>(Color(7, 7, 7));
    objects.add(make_shared<XZ_Rect>(123, 423, 147, 412, 554, light));

    auto center1 = Point3(400, 400, 200);
    auto center2 = center1 + Vec3(30,0,0);
    auto moving_sphere_material = make_shared<Lambertian>(Color(0.7, 0.3, 0.1));
    objects.add(make_shared<Moving_Sphere>(center1, center2, 0, 1, 50, moving_sphere_material));

    objects.add(make_shared<Sphere>(Point3(260, 150, 45), 50, make_shared<Dielectric>(1.5)));
    objects.add(make_shared<Sphere>(Point3(0, 150, 145), 50, make_shared<Metal>(Color(0.8, 0.8, 0.9), 1.0)));

    auto boundary = make_shared<Sphere>(Point3(360,150,145), 70, make_shared<Dielectric>(1.5));
    objects.add(boundary);
    objects.add(make_shared<Constant_Medium>(boundary, 0.2, Color(0.2, 0.4, 0.9)));
    boundary = make_shared<Sphere>(Point3(0, 0, 0), 5000, make_shared<Dielectric>(1.5));
    objects.add(make_shared<Constant_Medium>(boundary, .0001, Color(1,1,1)));

    auto emat = make_shared<Lambertian>(make_shared<Image_Text>("textures/earthmap.jpg"));
    objects.add(make_shared<Sphere>(Point3(400,200,400), 100, emat));
    auto pertext = make_shared<Noise_Text>(0.1);
    objects.add(make_shared<Sphere>(Point3(220,280,300), 80, make_shared<Lambertian>(pertext)));

    Hittable_List boxes2;
    auto white = make_shared<Lambertian>(Color(.73, .73, .73));
    int ns = 1000;
    for (int j = 0; j < ns; j++) {
        boxes2.add(make_shared<Sphere>(Point3::random(0,165), 10, white));
    }

    objects.add(make_shared<Translate>(make_shared<Rotate_Y>(make_shared<Bvh_Node>(boxes2, 0.0, 1.0), 15),Vec3(-100,270,395)));

    return objects;
}

Camera final_scene_cam(float aspect_ratio)
{
    auto lookfrom = Point3(478, 278, -600);
    auto lookat = Point3(278, 278, 0);
    auto vup = Vec3(0, 1, 0);
    float fov = 40.0;
    float dist_to_focus = 10.0;
    float aperture = 0;

    return Camera(lookfrom, lookat, vup, fov, aspect_ratio, aperture, dist_to_focus, 0.0, 1.0);
}