/*
 * Main program: Create a random world of Balls
 */

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <chrono>
#include <cstdlib>
#include "shader_consts.h"
#include "render.h"
#include "worlds.h"

using namespace std::chrono;

int main(int argc, char **argv)
{
    auto program_start = high_resolution_clock::now();

    // Image
    //float aspect_ratio = 16.0f / 9.0f;
    float aspect_ratio;
    int max_depth = 50;
    int image_width;
    int samples_per_pixel;
    int image_height;
    int image;

    if (argc < 5) {
        image = 0; 
        image_width = 400;
        image_height = 225;
        samples_per_pixel = 20;
    } else {
        image = atoi(argv[1]);
        image_width = atoi(argv[2]);
        image_height = atoi(argv[3]);
        samples_per_pixel = atoi(argv[4]);
    }

    aspect_ratio = float(image_width)/ (float(image_height));


    auto create_time_start = high_resolution_clock::now();
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
            image_height = static_cast<int>(image_width / aspect_ratio);
            cam = cornell_box_cam(aspect_ratio);
            background = Color(0, 0, 0);
            break;
        
        case 7:
            world = cornell_smoke();
            aspect_ratio = 1.0;
            image_width = 300;
            image_height = static_cast<int>(image_width / aspect_ratio);
            cam = cornell_smoke_cam(aspect_ratio);
            background = Color(0, 0, 0);
            break;

        default:
        case 8:
            world = final_scene();
            aspect_ratio = 1.0;
            image_width = 400;
            image_height = static_cast<int>(image_width / aspect_ratio);
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
    //image_height = static_cast<int>(image_width / aspect_ratio);

    // Render output variable
    Vec3 *image_pixels = new Vec3[image_height * image_width * sizeof(Vec3)];

    auto create_time_end = high_resolution_clock::now();

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

    std::cerr << "\nRender complete and image saved." << std::endl;
    file.close();

    // Print timing of program
    // Total Time
    auto program_end = high_resolution_clock::now();

    // Creation Time
    auto create_time = duration_cast<milliseconds>(create_time_end - create_time_start);
    std::cout << "World Creation Time: " << create_time.count() << "ms" << std::endl;

    // Render Time
    auto render_time = duration_cast<milliseconds>(render_time_end - render_time_start);
    std::cout << "Render Time: " << render_time.count() << "ms" << std::endl;

    // Save image time
    auto save_time = duration_cast<milliseconds>(save_time_end - save_time_start);
    std::cout << "Image Save Time: " << save_time.count() << "ms" << std::endl;

    auto time = duration_cast<milliseconds>(program_end - program_start);
    std::cout << "Total Time: " << time.count() << "ms" << std::endl;

    return 0;
}