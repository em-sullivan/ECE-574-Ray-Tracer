/*
 * Main program: Create a random world of Balls
 */

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <chrono>
#include "shader_consts.h"
#include "render.h"
#include "worlds.h"

using namespace std::chrono;

int main(int argc, char **argv)
{
    int image;
    if (argc < 2) {
        image = 0;
    } else {
        std::stringstream str_to_int(argv[1]);
        str_to_int >> image;
    }

    auto program_start = high_resolution_clock::now();
    
    // Image
    float aspect_ratio = 16.0f / 9.0f;
    int image_width = 400;
    int samples_per_pixel = 500;
    int max_depth = 50;
    int image_height;

    // World
    Hittable_List world;
    Camera cam;
    Color background;
    
    switch(image) {
        case 0:
            // Generate the earth 
            world = earf();
            cam = earf_cam(aspect_ratio);
            background = Color(0.70, 0.80, 1.00);
            break;
            
        case 1:
            // Generate three balls
            world = three_balls();
            cam = three_balls_cam(aspect_ratio); 
            background = Color(0.70, 0.80, 1.00);
            break;

        case 2:
            // Generate random balls - takes a while!
            world = random_balls();
            cam = random_balls_cam(aspect_ratio);
            background = Color(0.70, 0.80, 1.00);
            break;

        case 3:
            // Generates two fuzzy balls
            world = two_fuzzy_balls();
            cam = two_fuzzy_balls_cam(aspect_ratio);
            background = Color(0.70, 0.80, 1.00);
            break;
            

        case 4:
            world = two_bit_balls();
            cam = two_bit_balls_cam(aspect_ratio);
            background = Color(0.70, 0.80, 1.00);
            break;


        case 5:
            world = simple_light();
            cam = simple_light_cam(aspect_ratio);
            background = Color(0, 0, 0);
            break;

        case 6:
            world = cornell_box();
            aspect_ratio = 1.0;
            image_width = 300;
            cam = cornell_box_cam(aspect_ratio);
            background = Color(0, 0, 0);
            break;
        
        case 7:
            world = cornell_smoke();
            aspect_ratio = 1.0;
            image_width = 300;
            cam = cornell_smoke_cam(aspect_ratio);
            background = Color(0, 0, 0);
            break;

        default:
        case 8:
            world = final_scene();
            aspect_ratio = 1.0;
            image_width = 400;
            cam = final_scene_cam(aspect_ratio);
            background = Color(0,0,0);
            break;

        case 9:
            world = solar_system();
            cam = solar_system_cam(aspect_ratio);
            background = Color(0, 0, 0);
            break;

        case 10:
            world = glow_balls();
            cam = glow_balls_cam(aspect_ratio);
            background = Color(0, 0, 0);
    }

    // Image height based on aspect ratio
    image_height = static_cast<int>(image_width / aspect_ratio);

    // Render output variable
    Vec3 *image_pixels = new Vec3[image_height * image_width * sizeof(Vec3)];

    // Output File
    std::fstream file;
    file.open("out.ppm", std::ios::out);
    
    // Render
    auto render_time_start = high_resolution_clock::now();
    render(image_pixels, image_height, image_width, samples_per_pixel, max_depth,
        cam, world, background);
    auto render_time_end = high_resolution_clock::now();

    // Save image
    auto save_time_start = high_resolution_clock::now();
    saveImage(file, image_pixels, image_height, image_width, samples_per_pixel);
    auto save_time_end = high_resolution_clock::now();

    delete image_pixels;

    std::cerr << "\nDone" << std::endl;
    file.close();

    // Print timing of program
    // Total Time
    auto program_end = high_resolution_clock::now();
    auto time = duration_cast<milliseconds>(program_end - program_start);
    std::cout << "Time :" << time.count() << "ms" << std::endl;

    // Render Time
    auto render_time = duration_cast<milliseconds>(render_time_end - render_time_start);
    std::cout << "Render time : " << render_time.count() << "ms" << std::endl;

    // Save image time
    auto save_time = duration_cast<milliseconds>(save_time_end - save_time_start);
    std::cout << "Image save time " << save_time.count() << "ms" << std::endl;

    return 0;
}