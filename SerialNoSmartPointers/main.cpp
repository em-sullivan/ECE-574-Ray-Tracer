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

Hittable *earth() {
    Material *mat =  new Lambertian(new Image_Text("textures/earthmap.jpg"));
    return new Sphere(Vec3(0,0, 0), 2, mat);
}

Hittable *two_Spheres() {
    Texture *checker = new Checkered( new Solid_Color(Vec3(0.2,0.3, 0.1)), new Solid_Color(Vec3(0.9, 0.9, 0.9)));
    int n = 50;
    Hittable **list = new Hittable*[n+1];
    list[0] =  new Sphere(Vec3(0,-10, 0), 10, new Lambertian( checker));
    list[1] =  new Sphere(Vec3(0, 10, 0), 10, new Lambertian( checker));

    return new Hittable_List(list,2);
}

Hittable *final() {
    int nb = 20;
    Hittable **list = new Hittable*[30];
    Hittable **Boxlist = new Hittable*[10000];
    Hittable **Boxlist2 = new Hittable*[10000];
    Material *white = new Lambertian( new Solid_Color(Vec3(0.73, 0.73, 0.73)) );
    Material *ground = new Lambertian( new Solid_Color(Vec3(0.48, 0.83, 0.53)) );
    int b = 0;
    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < nb; j++) {
            float w = 100;
            float x0 = -1000 + i*w;
            float z0 = -1000 + j*w;
            float y0 = 0;
            float x1 = x0 + w;
            float y1 = 100*(random_float()+0.01);
            float z1 = z0 + w;
            Boxlist[b++] = new Box(Vec3(x0,y0,z0), Vec3(x1,y1,z1), ground);
        }
    }
    int l = 0;
    list[l++] = new Bvh_Node(Boxlist, 0, b, 0, 1);
    Material *light = new Diffuse_Light( new Solid_Color(Vec3(7, 7, 7)) );
    list[l++] = new XZ_Rect(123, 423, 147, 412, 554, light);
    Vec3 center(400, 400, 200);
    list[l++] = new Moving_Sphere(center, center+Vec3(30, 0, 0), 0, 1, 50, new Lambertian(new Solid_Color(Vec3(0.7, 0.3, 0.1))));
    list[l++] = new Sphere(Vec3(260, 150, 45), 50, new Dielectric(1.5));
    list[l++] = new Sphere(Vec3(0, 150, 145), 50, new Metal(Vec3(0.8, 0.8, 0.9), 10.0));
    Hittable *boundary = new Sphere(Vec3(360, 150, 145), 70, new Dielectric(1.5));
    list[l++] = boundary;
    list[l++] = new Constant_Medium(boundary, 0.2, new Solid_Color(Vec3(0.2, 0.4, 0.9)));
    boundary = new Sphere(Vec3(0, 0, 0), 5000, new Dielectric(1.5));
    list[l++] = new Constant_Medium(boundary, 0.0001, new Solid_Color(Vec3(1.0, 1.0, 1.0)));
    Material *emat =  new Lambertian(new Image_Text("textures/earthmap.jpg"));
    list[l++] = new Sphere(Vec3(400,200, 400), 100, emat);
    Texture *pertext = new Noise_Text(0.1);
    list[l++] =  new Sphere(Vec3(220,280, 300), 80, new Lambertian( pertext ));
    int ns = 1000;
    for (int j = 0; j < ns; j++) {
        Boxlist2[j] = new Sphere(Vec3(165*random_float(), 165*random_float(), 165*random_float()), 10, white);
    }
    list[l++] =   new Translate(new Rotate_Y(new Bvh_Node(Boxlist2, 0, ns, 0.0, 1.0), 15), Vec3(-100,270,395));
    return new Hittable_List(list,l);
}

Hittable *cornell_final() {
    Hittable **list = new Hittable*[30];
    Hittable **Boxlist = new Hittable*[10000];
    Texture *pertext = new Noise_Text(0.1);  
    Material *mat =  new Lambertian(new Image_Text("textures/earthmap.jpg"));
    int i = 0;
    Material *red = new Lambertian( new Solid_Color(Vec3(0.65, 0.05, 0.05)) );
    Material *white = new Lambertian( new Solid_Color(Vec3(0.73, 0.73, 0.73)) );
    Material *green = new Lambertian( new Solid_Color(Vec3(0.12, 0.45, 0.15)) );
    Material *light = new Diffuse_Light( new Solid_Color(Vec3(7, 7, 7)) );
    //list[i++] = new Sphere(Vec3(260, 50, 145), 50,mat);
    list[i++] = new YZ_Rect(0, 555, 0, 555, 555, green);
    list[i++] = new YZ_Rect(0, 555, 0, 555, 0, red);
    list[i++] = new XZ_Rect(113, 443, 127, 432, 554, light);
    list[i++] = new XZ_Rect(0, 555, 0, 555, 555, white);
    list[i++] = new XZ_Rect(0, 555, 0, 555, 0, white);
    list[i++] = new XY_Rect(0, 555, 0, 555, 555, white);
    /*
    Hittable *boundary = new Sphere(Vec3(160, 50, 345), 50, new Dielectric(1.5));
    list[i++] = boundary;
    list[i++] = new Constant_Medium(boundary, 0.2, new Solid_Color(Vec3(0.2, 0.4, 0.9)));
    list[i++] = new Sphere(Vec3(460, 50, 105), 50, new Dielectric(1.5));
    list[i++] = new Sphere(Vec3(120, 50, 205), 50, new Lambertian(pertext));
    int ns = 10000;
    for (int j = 0; j < ns; j++) {
        Boxlist[j] = new Sphere(Vec3(165*random_float(), 330*random_float(), 165*random_float()), 10, white);
    }
    list[i++] =   new Translate(new Rotate_Y(new Bvh_Node(Boxlist,ns, 0.0, 1.0), 15), Vec3(265,0,295));
    */
    Hittable *boundary2 = new Translate(new Rotate_Y(new Box(Vec3(0, 0, 0), Vec3(165, 165, 165), new Dielectric(1.5)), -18), Vec3(130,0,65));
    list[i++] = boundary2;
    list[i++] = new Constant_Medium(boundary2, 0.2, new Solid_Color(Vec3(0.9, 0.9, 0.9)));
    return new Hittable_List(list,i);
}

Hittable *cornell_balls() {
    Hittable **list = new Hittable*[9];
    int i = 0;
    Material *red = new Lambertian( new Solid_Color(Vec3(0.65, 0.05, 0.05)) );
    Material *white = new Lambertian( new Solid_Color(Vec3(0.73, 0.73, 0.73)) );
    Material *green = new Lambertian( new Solid_Color(Vec3(0.12, 0.45, 0.15)) );
    Material *light = new Diffuse_Light( new Solid_Color(Vec3(5, 5, 5)) );
    list[i++] = new YZ_Rect(0, 555, 0, 555, 555, green);
    list[i++] = new YZ_Rect(0, 555, 0, 555, 0, red);
    list[i++] = new XZ_Rect(113, 443, 127, 432, 554, light);
    list[i++] = new XZ_Rect(0, 555, 0, 555, 555, white);
    list[i++] = new XZ_Rect(0, 555, 0, 555, 0, white);
    list[i++] = new XY_Rect(0, 555, 0, 555, 555, white);
    Hittable *boundary = new Sphere(Vec3(160, 100, 145), 100, new Dielectric(1.5));
    list[i++] = boundary;
    list[i++] = new Constant_Medium(boundary, 0.1, new Solid_Color(Vec3(1.0, 1.0, 1.0)));
    list[i++] = new Translate(new Rotate_Y(new Box(Vec3(0, 0, 0), Vec3(165, 330, 165), white),  15), Vec3(265,0,295));
    return new Hittable_List(list,i);
}

Hittable *cornell_smoke() {
    Hittable **list = new Hittable*[8];
    int i = 0;
    Material *red = new Lambertian( new Solid_Color(Vec3(0.65, 0.05, 0.05)) );
    Material *white = new Lambertian( new Solid_Color(Vec3(0.73, 0.73, 0.73)) );
    Material *green = new Lambertian( new Solid_Color(Vec3(0.12, 0.45, 0.15)) );
    Material *light = new Diffuse_Light( new Solid_Color(Vec3(7, 7, 7)) );
    list[i++] = new YZ_Rect(0, 555, 0, 555, 555, green);
    list[i++] = new YZ_Rect(0, 555, 0, 555, 0, red);
    list[i++] = new XZ_Rect(113, 443, 127, 432, 554, light);
    list[i++] = new XZ_Rect(0, 555, 0, 555, 555, white);
    list[i++] = new XZ_Rect(0, 555, 0, 555, 0, white);
    list[i++] = new XY_Rect(0, 555, 0, 555, 555, white);

    Hittable *b1 = new Translate(new Rotate_Y(new Box(Vec3(0, 0, 0), Vec3(165, 165, 165), white), -18), Vec3(130,0,65));
    Hittable *b2 = new Translate(new Rotate_Y(new Box(Vec3(0, 0, 0), Vec3(165, 330, 165), white),  15), Vec3(265,0,295));
    list[i++] = new Constant_Medium(b1, 0.01, new Solid_Color(Vec3(1.0, 1.0, 1.0)));
    list[i++] = new Constant_Medium(b2, 0.01, new Solid_Color(Vec3(0.0, 0.0, 0.0)));
    return new Hittable_List(list,i);
}

Hittable *cornell_Box() {
    Hittable **list = new Hittable*[8];
    int i = 0;
    Material *red = new Lambertian( new Solid_Color(Vec3(0.65, 0.05, 0.05)) );
    Material *white = new Lambertian( new Solid_Color(Vec3(0.73, 0.73, 0.73)) );
    Material *green = new Lambertian( new Solid_Color(Vec3(0.12, 0.45, 0.15)) );
    Material *light = new Diffuse_Light( new Solid_Color(Vec3(15, 15, 15)) );
    list[i++] = new YZ_Rect(0, 555, 0, 555, 555, green);
    list[i++] = new YZ_Rect(0, 555, 0, 555, 0, red);
    list[i++] = new XZ_Rect(113, 443, 127, 432, 554, light);
    list[i++] = new XZ_Rect(0, 555, 0, 555, 555, white);
    list[i++] = new XZ_Rect(0, 555, 0, 555, 0, white);
    list[i++] = new XY_Rect(0, 555, 0, 555, 555, white);
    list[i++] = new Translate(new Rotate_Y(new Box(Vec3(0, 0, 0), Vec3(165, 165, 165), white), -18), Vec3(130,0,65));
    list[i++] = new Translate(new Rotate_Y(new Box(Vec3(0, 0, 0), Vec3(165, 330, 165), white),  15), Vec3(265,0,295));
    return new Hittable_List(list,i);
}

Hittable *two_perlin_Spheres() {
    Texture *pertext = new Noise_Text(4);
    Hittable **list = new Hittable*[2];
    list[0] =  new Sphere(Vec3(0,-1000, 0), 1000, new Lambertian( pertext ));
    list[1] =  new Sphere(Vec3(0, 2, 0), 2, new Lambertian( pertext ));
    return new Hittable_List(list,2);
}

Hittable *simple_light() {
    Texture *pertext = new Noise_Text(4);
    Hittable **list = new Hittable*[4];
    list[0] =  new Sphere(Vec3(0,-1000, 0), 1000, new Lambertian(pertext));
    list[1] =  new Sphere(Vec3(0, 2, 0), 2, new Lambertian( pertext ));
    list[2] =  new Sphere(Vec3(0, 7, 0), 2, new Diffuse_Light( new Solid_Color(Vec3(4, 4, 4))));
    list[3] =  new XY_Rect(3, 5, 1, 3, -2, new Diffuse_Light(new Solid_Color(Vec3(4, 4, 4))));
    return new Hittable_List(list,4);
}

Hittable *random_balls() {
    int n = 50000;
    Hittable **list = new Hittable*[n+1];
    auto ground_material = new Lambertian(Color(0.5, 0.5, 0.5));
    list[0] = new Sphere(Point3(0, -1000, 0), 1000, ground_material);
    int i = 1;
    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
           
            float choose_mat = random_float();
            Point3 center(a+0.9*random_float(),0.2, b+0.9*random_float()); 
            
            if ((center-Point3(4,0.2,0)).length() > 0.9) { 
                
                Material *sphere_material;
                if (choose_mat < 0.8) {  // diffuse
                    auto albedo = Color::random() * Color::random();
                    sphere_material = new Lambertian(albedo);
                    list[i++]= new Sphere(center, 0.2, sphere_material);
                }
                else if (choose_mat < 0.95) { // Metal
                    auto albedo = Color::random(0.5, 1);
                    auto fuzz = random_float(0, 0.5);
                    sphere_material = new Metal(albedo, fuzz);
                    list[i++] = new Sphere(center, 0.2, sphere_material);
                }
                else {  // glass
                    list[i++] = new Sphere(center, 0.2, new Dielectric(1.5));
                }
            }
        }
    }

    // Big balls
    list[i++] = new Sphere(Vec3(0, 1, 0), 1.0, new Dielectric(1.5));
    list[i++] = new Sphere(Vec3(-4, 1, 0), 1.0, new Lambertian(new Solid_Color(Vec3(0.4, 0.2, 0.1))));
    list[i++] = new Sphere(Vec3(4, 1, 0), 1.0, new Metal(Vec3(0.7, 0.6, 0.5), 0.0));

    //return new Hittable_List(list,i);
    return new Bvh_Node(list, 0, i, 0.0, 1.0);
}

Color ray_color(const Ray &r, const Color& background, const Hittable *world, int depth)
{
    hit_record rec;

    // If the ray bounce limit is exceeded, no more light is gathered
    if (depth <= 0)
        return Color(0, 0, 0);

    // If the ray didn't hit anything, return the background
    if (!world->hit(r, 0.001f, INF, rec)) {
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
    int samples_per_pixel = 20;
    int max_depth = 50;
    int image_height;

    // World
   Hittable *world;
    Point3 lookfrom;
    Point3 lookat;
    Point3 vup;
    Color background;
    float fov;
    float aperture;
    float dist_to_focus;


    switch(image) {
        case 1:
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

        case 2:
            world = simple_light();
            lookfrom = Point3(26, 3, 6);
            lookat = Point3(0, 2, 0);
            vup = Vec3(0, 1, 0);
            background = Color(0, 0, 0);
            fov = 20;
            dist_to_focus = 10.0;
            aperture = 0;
            break;

        case 3:
            world = cornell_final();
            aspect_ratio = 1.0;
            image_width = 300;
            lookfrom = Point3(278, 278, -800);
            lookat = Point3(278, 278, 0);
            vup = Vec3(0, 1, 0);
            fov = 40;
            dist_to_focus = 10.0;
            aperture = 0;
            break;
        
        case 4:
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
        
        case 5:
            world = two_perlin_Spheres();
            lookfrom = Point3(13, 2, 3);
            lookat = Point3(0, 0, 0);
            vup = Vec3(0, 1, 0);
            background = Color(0.70, 0.80, 1.00);
            fov = 20;
            dist_to_focus = 10.0;
            aperture = 0;
            break;
        
        case 6:
            world = two_Spheres();
            lookfrom = Point3(13, 2, 3);
            lookat = Point3(0, 0, 0);
            vup = Vec3(0, 1, 0);
            background = Color(0.70, 0.80, 1.00);
            fov = 20;
            dist_to_focus = 10.0;
            aperture = 0;
            break;

        default:
        case 7:
            world = final();
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