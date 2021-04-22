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
    float fov = 20;
    float dist_to_focus = 10.0;
    float aperture = 0;

    return Camera(lookfrom, lookat, vup, fov, aspect_ratio, aperture, dist_to_focus, 0.0, 1.0);
}

Hittable_List earf()
{
    auto earf_texture = make_shared<Image_Text>("../Common/textures/earthmap.jpg");
    auto earf_surface = make_shared<Lambertian>(earf_texture);
    auto globe = make_shared<Sphere>(Point3(0, 0, 0), 2, earf_surface);

    return Hittable_List(globe);
}

Camera earf_cam(float aspect_ratio)
{
    auto lookfrom = Point3(13, 2, 3);
    auto lookat = Point3(0, 0, 0);
    auto vup = Vec3(0, 1, 0);
    float fov = 20;
    float dist_to_focus = 10.0;
    float aperture = 0;

    return Camera(lookfrom, lookat, vup, fov, aspect_ratio, aperture, dist_to_focus, 0.0, 1.0);
}

Hittable_List pool_table()
{
    Hittable_List table;

    //800x300 image

    auto ball1_texture = make_shared<Image_Text>("../Common/textures/Pool Ball Skins/Ball1.jpg");
    auto ball2_texture = make_shared<Image_Text>("../Common/textures/Pool Ball Skins/Ball2.jpg");
    auto ball3_texture = make_shared<Image_Text>("../Common/textures/Pool Ball Skins/Ball3.jpg");
    auto ball4_texture = make_shared<Image_Text>("../Common/textures/Pool Ball Skins/Ball4.jpg");
    auto ball5_texture = make_shared<Image_Text>("../Common/textures/Pool Ball Skins/Ball5.jpg");
    auto ball6_texture = make_shared<Image_Text>("../Common/textures/Pool Ball Skins/Ball6.jpg");
    auto ball7_texture = make_shared<Image_Text>("../Common/textures/Pool Ball Skins/Ball7.jpg");
    auto ball8_texture = make_shared<Image_Text>("../Common/textures/Pool Ball Skins/Ball8.jpg");
    auto ball9_texture = make_shared<Image_Text>("../Common/textures/Pool Ball Skins/Ball9.jpg");
    auto ball10_texture = make_shared<Image_Text>("../Common/textures/Pool Ball Skins/Ball10.jpg");
    auto ball11_texture = make_shared<Image_Text>("../Common/textures/Pool Ball Skins/Ball11.jpg");
    auto ball12_texture = make_shared<Image_Text>("../Common/textures/Pool Ball Skins/Ball12.jpg");
    auto ball13_texture = make_shared<Image_Text>("../Common/textures/Pool Ball Skins/Ball13.jpg");
    auto ball14_texture = make_shared<Image_Text>("../Common/textures/Pool Ball Skins/Ball14.jpg");
    auto ball15_texture = make_shared<Image_Text>("../Common/textures/Pool Ball Skins/Ball15.jpg");
    auto ball_cue_texture = make_shared<Image_Text>("../Common/textures/Pool Ball Skins/BallCue.jpg");

    auto ball1_surface = make_shared<Lambertian>(ball1_texture);
    auto ball2_surface = make_shared<Lambertian>(ball2_texture);
    auto ball3_surface = make_shared<Lambertian>(ball3_texture);
    auto ball4_surface = make_shared<Lambertian>(ball4_texture);
    auto ball5_surface = make_shared<Lambertian>(ball5_texture);
    auto ball6_surface = make_shared<Lambertian>(ball6_texture);
    auto ball7_surface = make_shared<Lambertian>(ball7_texture);
    auto ball8_surface = make_shared<Lambertian>(ball8_texture);
    auto ball9_surface = make_shared<Lambertian>(ball9_texture);
    auto ball10_surface = make_shared<Lambertian>(ball10_texture);
    auto ball11_surface = make_shared<Lambertian>(ball11_texture);
    auto ball12_surface = make_shared<Lambertian>(ball12_texture);
    auto ball13_surface = make_shared<Lambertian>(ball13_texture);
    auto ball14_surface = make_shared<Lambertian>(ball14_texture);
    auto ball15_surface = make_shared<Lambertian>(ball15_texture);
    auto ball_cue_surface = make_shared<Lambertian>(ball_cue_texture);
    // auto table = make_shared<Sphere>(Point3(0, 0, 0), 2, ball1_surface);
    table.add(make_shared<Sphere>(Point3(-.05, 0,    2), .5, ball1_surface)); //For Point3(diagonal, y, x)
    table.add(make_shared<Sphere>(Point3(-.1, .5,   2.85), .5, ball2_surface));
    table.add(make_shared<Sphere>(Point3(-.1, -.5,  2.85), .5, ball3_surface));
    table.add(make_shared<Sphere>(Point3(-.15, -1,   3.7), .5, ball4_surface));
    table.add(make_shared<Sphere>(Point3(-.15, 0,    3.7), .5, ball5_surface));
    table.add(make_shared<Sphere>(Point3(-.15, 1,    3.7), .5, ball6_surface));
    table.add(make_shared<Sphere>(Point3(-.185, 1.5,  4.55), .5, ball7_surface));
    table.add(make_shared<Sphere>(Point3(-.185, .5,   4.55), .5, ball8_surface));
    table.add(make_shared<Sphere>(Point3(-.185, -.5,  4.55), .5, ball9_surface));
    table.add(make_shared<Sphere>(Point3(-.185, -1.5, 4.55), .5, ball10_surface));
    table.add(make_shared<Sphere>(Point3(-.25, -2.0, 5.40), .5, ball11_surface));
    table.add(make_shared<Sphere>(Point3(-.25, -1.0, 5.40), .5, ball12_surface));
    table.add(make_shared<Sphere>(Point3(-.25, 0.0,  5.40), .5, ball13_surface));
    table.add(make_shared<Sphere>(Point3(-.25, 1.0,  5.40), .5, ball14_surface));
    table.add(make_shared<Sphere>(Point3(-.25, 2.0,  5.40), .5, ball15_surface));

    table.add(make_shared<Sphere>(Point3(0, 0, -5), .5, ball_cue_surface));

    //Ground
    auto ground = make_shared<Checkered>(Color(54./255, 120./255, 75./255), Color(54./255, 120./255, 75./255));
    table.add(make_shared<Sphere>(Point3(-100.5, 0.0, -1.0), 100.0, make_shared<Lambertian>(ground)));    

    //Light for shading
    // auto difflight = make_shared<Diffuse_Light>(Color(5, 5, 5));
    // table.add(make_shared<Sphere>(Point3(3, 10, 0), 3, difflight));
    auto difflight2 = make_shared<Diffuse_Light>(Color(5, 5, 5));
    table.add(make_shared<XZ_Rect>(-200, 200, -100, 654, 554, difflight2));
    // table.add(make_shared<Sphere>(Point3(2, -1, 2), .3, difflight2));
    // table.add(make_shared<Sphere>(Point3(2, -.5, 4.55), .3, difflight2));
    // table.add(make_shared<Sphere>(Point3(2, -2.75, 1), .3, difflight2));
    // table.add(make_shared<Sphere>(Point3(2, 2.75, -1), .3, difflight2));
    // table.add(make_shared<Sphere>(Point3(2, 0, -1.7), .3, difflight2));
    // table.add(make_shared<Sphere>(Point3(2, 1, 2.4), .3, difflight2));
    // table.add(make_shared<Sphere>(Point3(2, 1.7, 2.1), .3, difflight2));
    // table.add(make_shared<Sphere>(Point3(2, -1.7, .67), .3, difflight2));

    return table;
}

Camera pool_table_cam(float aspect_ratio)
{
    // auto lookfrom = Point3(7, 0, -20);
    // auto lookat = Point3(-2, 0, 10);
    auto lookfrom = Point3(13, 2, 3);
    auto lookat = Point3(0, 0, 0);
    auto vup = Vec3(0, 1, 0);
    float fov = 25;
    // float dist_to_focus = 15.0;
    float dist_to_focus = (lookfrom-lookat).length();
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

Hittable_List solar_system()
{
    Hittable_List world;

    // Star colors
    auto star1 = make_shared<Solid_Color>(Color(1, 1, 1)); // White
    auto star2 = make_shared<Solid_Color>(Color(0.75, 0.6, 0.5)); // Yellow
    auto star3 = make_shared<Solid_Color>(Color(0.93, 0.41, 0.24)); // Red
    auto star4 = make_shared<Solid_Color>(Color(0.4, 0.82, 0.95)); // Blue

    // Create Sun and slightly bigger light source
    auto sun_texture = make_shared<Image_Text>("../Common/textures/sunHQ.jpg");
    auto sun_surface = make_shared<Diffuse_Light>(sun_texture);
    world.add(make_shared<Sphere>(Point3(0, 0, -320), 300.0, sun_surface));
    auto light = make_shared<Diffuse_Light>(Color(4, 4, 4));
    world.add(make_shared<Sphere>(Point3(0, 0, -1300), 600.0, light));

    // Create each planet in line
    world.add(make_shared<Sphere>(Point3(0, 0, -10), 2,
        make_shared<Lambertian>(make_shared<Image_Text>("../Common/textures/mercury.jpg"))));
    
    world.add(make_shared<Sphere>(Point3(0, 0, 0), 3.6,
        make_shared<Lambertian>(make_shared<Image_Text>("../Common/textures/venus.jpg"))));

    world.add(make_shared<Sphere>(Point3(0, 0, 13), 4.4,
        make_shared<Lambertian>(make_shared<Image_Text>("../Common/textures/earth.jpg"))));

    world.add(make_shared<Sphere>(Point3(0, 0, 27), 2.4,
        make_shared<Lambertian>(make_shared<Image_Text>("../Common/textures/mars.jpg"))));

    world.add(make_shared<Sphere>(Point3(0, 0, 80), 34.4,
        make_shared<Lambertian>(make_shared<Image_Text>("../Common/textures/jupiterHQ.jpg"))));

    world.add(make_shared<Sphere>(Point3(0, 0, 190), 28.0,
        make_shared<Lambertian>(make_shared<Image_Text>("../Common/textures/saturnHQ.jpg"))));

    world.add(make_shared<Sphere>(Point3(0, 0, 310), 16.4,
        make_shared<Lambertian>(make_shared<Image_Text>("../Common/textures/uranus.jpg"))));

    world.add(make_shared<Sphere>(Point3(0, 0, 450), 16.0,
        make_shared<Lambertian>(make_shared<Image_Text>("../Common/textures/neptune.jpg"))));
    
    world.add(make_shared<Sphere>(Point3(0, 0, 575), 2.75,
        make_shared<Lambertian>(make_shared<Image_Text>("../Common/textures/pluto.jpg"))));

    // Generate random stars in the background
    for (int a = -450; a < 450; a+=20) {
        for (int c = -20; c < 1100; c+=20) {
            float star_color = random_float();

            float rand1 = random_float();
            rand1 *= (20.f + 0.999999f);
            rand1 = truncf(rand1);

            float rand2 = random_float();
            rand2 *= (20.f + 0.999999f);
            rand2 = truncf(rand2);

            float rand3 = random_float();
            rand3 *= (20.f + 0.999999f);
            rand3 = truncf(rand3);

            Vec3 center(250 + rand1 + (800 - c), a + rand2, c + rand3);
             if (star_color < 0.7f) {
                world.add(make_shared<Sphere>(center, random_float(), 
                    make_shared<Diffuse_Light>(star1)));
            } else if  (star_color < 0.9f) {
                world.add(make_shared<Sphere>(center, random_float(), 
                    make_shared<Diffuse_Light>(star2)));
            } else if  (star_color < 0.95f) {
                world.add(make_shared<Sphere>(center, random_float(), 
                    make_shared<Diffuse_Light>(star3)));
            } else {
                world.add(make_shared<Sphere>(center, random_float(), 
                    make_shared<Diffuse_Light>(star4)));
            }
        }
    }
    return world;
}

Camera solar_system_cam(float aspect_ratio)
{
    auto lookfrom = Point3(-145, 0, -25);
    auto lookat = Point3(-110, 0, 5);
    auto vup = Vec3(0, 1, 0);
    float fov = 52.0;
    float dist_to_focus = 100.0;
    float aperture = 0.1;

    return Camera(lookfrom, lookat, vup, fov, aspect_ratio, aperture, dist_to_focus, 0.0, 1.0);
}

Hittable_List glow_balls()
{
    Hittable_List world;

    // Gray ground
    auto ground_material = make_shared<Lambertian>(Color(0.5, 0.5, 0.5));
    world.add(make_shared<Sphere>(Point3(0, -1000, 0), 1000, ground_material));

    for (int a = -8; a < 8; a++) {
        for (int b = -8; b < 8; b++) {

            // "Randomly" choose material and point
            auto choose_mat = random_float();
            Point3 center(a + 0.9 * random_float(), 0.2, b + 0.9 * random_float());

            if ((center - Point3(4, 0.2, 0)).length() > 0.9) {
                shared_ptr<Material> sphere_material;
                // Glow ball
                auto albedo = Color::random() * Color::random();
                albedo[1] = 0;
                sphere_material = make_shared<Diffuse_Light>(albedo);

                if (choose_mat < .5) {
                    world.add(make_shared<Sphere>(center, 0.2, sphere_material));
                } else {
                    world.add(make_shared<Moving_Sphere>(center, 
                        center + Point3(0, 0, random_float()),
                        0.0, 1.0, 0.2, sphere_material));
                }
            }

        }
    }

    // Big old mirror ball
    auto ball1 =make_shared<Dielectric>(1.5);
    world.add(make_shared<Sphere>(Point3(4, 1, 1.5), 1.0, ball1));
    
    return world;
}

Camera glow_balls_cam(float aspect_ratio)
{
    Point3 lookfrom = Point3(13, 2, 5);
    Point3 lookat = Point3(0, 0, 0);
    Point3 vup = Vec3(0, 1, 0);
    float fov = 25;
    float dist_to_focus = 10.0;
    float aperture = .1;

    return Camera(lookfrom, lookat, vup, fov, aspect_ratio, aperture, dist_to_focus, 0.0, 1.0);
}