#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <chrono>
#include <cstdlib>
#include <curand_kernel.h>
#include "Vec3.h"
#include "Color.h"
#include "Ray.h"
#include "Sphere.h"
#include "Hittable_List.h"
#include "Camera.h"
#include "Texture.h"
#include "Render.h"
#include "Moving-Sphere.h"

#include "shader_stb_image.h"

using namespace std::chrono;

#define RND (curand_uniform(&local_rand_state))

__global__ void glow_balls(Hittable **d_list, Hittable **d_world, Camera **d_camera, int nx, int ny, curandState *rand_state)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;
        int i = 0;
        d_list[i++] = new Sphere(Vec3(0, -1000, -1), 1000, new Lambertian(new Solid_Color(0.5, 0.5, 0.5)));
        
        for (int a = -8; a < 8; a++) {
            for (int b = -8; b < 8; b++) {
                float choose_mat = RND;
                Vec3 center(a + 0.9f * RND, 0.2, b + 0.9f * RND);
                if (choose_mat < .5f) {
                    d_list[i++] = new Sphere(center, 0.2, 
                        new Diffuse_Light(new Solid_Color(RND * RND, 0, RND * RND)));
                } else {
                    d_list[i++] = new Moving_Sphere(center, center + Vec3(0, 0, RND), 0.0, 1.0, 0.2,
                        new Diffuse_Light(new Solid_Color(RND * RND, 0, RND * RND)));
                }

            }
        }

        d_list[i++] = new Sphere(Vec3(4, 1, 1.5f), 1.0, new Dielectric(1.5));

        // World
        *rand_state = local_rand_state;
        *d_world = new Hittable_List(d_list, 16 * 16 + 2);

        // Camera
        Vec3 lookfrom = Vec3(13, 2, 5);
        Vec3 lookat = Vec3(0, 0, 0);
        float dist_to_focus = 10.0;
        float aperture = .1f;
        *d_camera = new Camera(lookfrom, lookat, Vec3(0,1,0), 25.0, float(nx)/float(ny), aperture, dist_to_focus, 0 ,1);
    }
}

__global__ void free_glow_balls(Hittable **d_list, Hittable **d_world, Camera **d_camera) 
{
    for(int i=0; i < (16 * 16 + 2); i++) {
        delete ((Hittable *)d_list[i])->mat_ptr;
        delete d_list[i];
    }
    delete *d_world;
    delete *d_camera;
}

__global__ void create_world(Hittable **d_list, Hittable **d_world, Camera **d_camera, int nx, int ny, curandState *rand_state) 
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;
        d_list[0] = new Sphere(Vec3(0,-1000.0,-1), 1000, new Lambertian(new Solid_Color(Vec3(0.5,0.5,0.5))));
        int i = 1;
        for(int a = -11; a < 11; a++) {
            for(int b = -11; b < 11; b++) {
                float choose_mat = RND;
                Vec3 center(a+RND,0.2,b+RND);
                if(choose_mat < 0.8f) {
                    d_list[i++] = new Sphere(center, 0.2, new Lambertian(new Solid_Color(Vec3(RND*RND, RND*RND, RND*RND))));
                }
                else if(choose_mat < 0.95f) {
                    d_list[i++] = new Sphere(center, 0.2, new Metal(Vec3(0.5f*(1.0f+RND), 0.5f*(1.0f+RND), 0.5f*(1.0f+RND)), 0.5f*RND));
                }
                else {
                    d_list[i++] = new Sphere(center, 0.2, new Dielectric(1.5));
                }
            }
        }

        d_list[i++] = new Sphere(Vec3(0, 1,0),  1.0, new Dielectric(1.5));
        d_list[i++] = new Sphere(Vec3(-4, 1, 0), 1.0 , new Lambertian(new Solid_Color(Vec3(0.4,0.3,0.1))));
        //d_list[i++] =  new Sphere(Vec3(0, 4, 5),  1.0, new Diffuse_Light( new Solid_Color(Vec3(7, 7, 7))));
        d_list[i++] = new Sphere(Vec3(4, 1, 0),  1.0, new Metal(Vec3(0.7, 0.6, 0.5), 0.0));

        
        *rand_state = local_rand_state;
        *d_world  = new Hittable_List(d_list, 22*22+1+3);

        Vec3 lookfrom = Vec3(13,2,3);
        Vec3 lookat = Vec3(0,0,0);
        float dist_to_focus = 10.0;
        float aperture = 0.1;
        *d_camera   = new Camera(lookfrom, lookat, Vec3(0,1,0), 25.0, float(nx)/float(ny), aperture, dist_to_focus, 0 ,1);
    }
}

__global__ void free_world(Hittable **d_list, Hittable **d_world, Camera **d_camera) 
{
    for(int i=0; i < 22*22+1+3; i++) {
        delete ((Sphere *)d_list[i])->mat_ptr;
        delete d_list[i];
    }
    delete *d_world;
    delete *d_camera;
}

__global__ void solar_system(Hittable **d_list, Hittable **d_world, Camera **d_camera, int nx, int ny, curandState *rand_state, int tex_nx, int tex_ny, int texHQ_nx, int texHQ_ny, unsigned char *sun,  
                                              unsigned char *mercury, unsigned char *venus, unsigned char *earth,  unsigned char *mars,  unsigned char *jupiter,  unsigned char *saturn,  unsigned char *uranus,  unsigned char *neptune, unsigned char* pluto) 
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;
        int i = 0;
        Texture *sunText = new Image_Text(sun, texHQ_nx, texHQ_ny);
        Texture *star1Text = new Solid_Color(Vec3(1, 1, 1));                // White 
        Texture *star2Text = new Solid_Color(Vec3(0.75, 0.6, 0.5));     // Yellow
        Texture *star3Text = new Solid_Color(Vec3(0.93, 0.41, 0.24)); // Red
        Texture *star4Text = new Solid_Color(Vec3(0.4, .82, 0.95));    // Blue 

        // Create sun and slightly bigger light source
        d_list[i++] = new Sphere(Vec3(0, 0, -320), 300.0 , new Diffuse_Light(sunText));
        d_list[i++] = new Sphere(Vec3(0, 0, -1300), 600.0 , new Diffuse_Light(new Solid_Color(Vec3(0.25, 0.2, 0.12))));

        // Create each planet in a line
        d_list[i++] = new Sphere(Vec3(0, 0, -10), 2, new Lambertian(new Image_Text(mercury, tex_nx, tex_ny)));
        d_list[i++] = new Sphere(Vec3(0, 0, 0), 3.6, new  Lambertian(new Image_Text(venus, tex_nx, tex_ny)));
        d_list[i++] = new Sphere(Vec3(0, 0, 13), 4.4, new  Lambertian(new Image_Text(earth, tex_nx, tex_ny)));
        d_list[i++] = new Sphere(Vec3(0, 0, 27), 2.4, new  Lambertian(new Image_Text(mars, tex_nx, tex_ny)));
        d_list[i++] = new Sphere(Vec3(0, 0, 80), 34.0, new  Lambertian(new Image_Text(jupiter, texHQ_nx, texHQ_ny)));
        d_list[i++] = new Sphere(Vec3(0, 0, 190), 28.0, new  Lambertian(new Image_Text(saturn, texHQ_nx, texHQ_ny)));
        d_list[i++] = new Sphere(Vec3(0, 0, 310), 16.4 , new  Lambertian(new Image_Text(uranus, tex_nx, tex_ny)));
        d_list[i++] = new Sphere(Vec3(0, 0, 450),  16.0, new  Lambertian(new Image_Text(neptune, tex_nx, tex_ny)));
        d_list[i++] = new Sphere(Vec3(0, 0, 575),  2.75, new  Lambertian(new Image_Text(pluto, tex_nx, tex_ny)));


        // Generates random stars in the background
        // DEPENDS GREATLY on lookfrom, lookat, and fov
        for(int a = -450; a < 450; a+=20) {
                for(int c = -20; c < 1100; c+=20) { 
                    float starColor = RND;
                    
                    float rand1 = RND;
                    rand1 *= (20.f+0.999999f);
                    rand1 = truncf(rand1);
                    
                    float rand2 = RND;
                    rand2 *= (20.f+0.999999f);
                    rand2 = truncf(rand2);

                    float rand3 = RND;
                    rand3 *= (20.f+0.999999f);
                    rand3 = truncf(rand3);
                    
                    Vec3 center(250 + rand1 + (800 - c), a+rand2,  c+rand3);
                    if (starColor < 0.7f) {
                        d_list[i++] = new Sphere(center, RND, new Diffuse_Light(star1Text));
                    } else if  (starColor < 0.9f) {
                        d_list[i++] = new Sphere(center, RND, new Diffuse_Light(star2Text));
                    } else if  (starColor < 0.95f) {
                        d_list[i++] = new Sphere(center, RND, new Diffuse_Light(star3Text));
                    } else {
                        d_list[i++] = new Sphere(center, RND, new Diffuse_Light(star4Text));
                    }
                }
            }

        *rand_state = local_rand_state;
        *d_world  = new Hittable_List(d_list, 11+45*56);

        Vec3 lookfrom = Vec3(-145,0, -25);
        Vec3 lookat = Vec3(-110,0, 5);
        float dist_to_focus = 100.0;
        float aperture = 0.1;
        float fov = 52.0;
        *d_camera   = new Camera(lookfrom, lookat, Vec3(0,1,0), fov, float(nx)/float(ny), aperture, dist_to_focus, 0 ,1);
    }
}

__global__ void free_solar_system(Hittable **d_list, Hittable **d_world, Camera **d_camera) 
{
    for(int i=0; i < 11+45*56; i++) {
        delete ((Hittable *)d_list[i])->mat_ptr;
        delete d_list[i];
    }
    delete *d_world;
    delete *d_camera;
}

__global__ void pool_table(Hittable **d_list, Hittable **d_world, Camera **d_camera, int nx, int ny, curandState *rand_state) 
{

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;

        int i = 0;

        // Pool balls
        d_list[i++] = new Sphere(Vec3(-.05, 0, 2), .5,  new Metal(Vec3(4*0.8314, 4*0.83,  4*0.09), 1));
        d_list[i++] = new Sphere(Vec3(-.1, .5, 2.85), .5, new Metal(Vec3(4*0.059, 4*0.333, 4*0.694), 1));
        d_list[i++] = new Sphere(Vec3(-.1, -.5, 2.85), .5, new Metal(Vec3(4*0.73, 4*0.102, 4*0.102), 1));
        d_list[i++] = new Sphere(Vec3(-.15, -1, 3.7), .5, new Metal(Vec3(4*0.431, 4*0.102, 4*0.53), 1));
        d_list[i++] = new Sphere(Vec3(-.15, 0, 3.7), .5, new Metal(Vec3(0, 0, 0), 1));
        d_list[i++] = new Sphere(Vec3(-.15, 1, 3.7), .5, new Metal(Vec3(4*0.059, 4*0.302, 4*0.059), 1));
        d_list[i++] = new Sphere(Vec3(-.185, 1.5, 4.55), .5, new Metal(Vec3(4*0.37, 4*0.02,  4*0.01568), 1));
        d_list[i++] = new Sphere(Vec3(-.185, .5, 4.55), .5, new Metal(Vec3(4*0.80, 4*0.333, 4*0.063), 1));
        d_list[i++] = new Sphere(Vec3(-.185, -.5, 4.55), .5, new Metal(Vec3(4*0.8314, 4*0.83,  4*0.09), 1));
        d_list[i++] = new Sphere(Vec3(-.185, -1.5, 4.55), .5, new Metal(Vec3(4*0.059, 4*0.333, 4*0.694), 1));
        d_list[i++] = new Sphere(Vec3(-.25, -2.0, 5.40), .5, new Metal(Vec3(4*0.37, 4*0.02,  4*0.01568), 1));
        d_list[i++] = new Sphere(Vec3(-.25, -1.0, 5.40), .5, new Metal(Vec3(4*0.059, 4*0.302, 4*0.059), 1));
        d_list[i++] = new Sphere(Vec3(-.25, 0.0,  5.40), .5, new Metal(Vec3(4*0.80, 4*0.333, 4*0.063), 1));
        d_list[i++] = new Sphere(Vec3(-.25, 1.0, 5.40), .5, new Metal(Vec3(4*0.431, 4*0.102, 4*0.53), 1));
        d_list[i++] = new Sphere(Vec3(-.25, 2.0, 5.40), .5, new Metal(Vec3(4*0.73, 4*0.102, 4*0.102), 1));
        
        // Cue ball
        d_list[i++] = new Sphere(Vec3(0, 0, -5), .5, new Metal(Vec3(1, 1, 1), 1));

        // The ground of the table
        d_list[i++] = new Sphere(Vec3(-100.5, 0.0, -1.0), 100, new Lambertian(Vec3(.212, .4706, .294)));

        // Lighting above the table
        d_list[i++] = new Sphere(Vec3(100, 5, 0), 10, new Diffuse_Light(Vec3(20, 20, 20)));

        // Table
        *rand_state = local_rand_state;
        *d_world = new Hittable_List(d_list, 18);

        // Camera
        Vec3 lookfrom = Vec3(3, 0, -15);
        Vec3 lookat = Vec3(-2, 0, 10);
        float dist_to_focus = 25.0;
        float aperture = 0;
        *d_camera = new Camera(lookfrom, lookat, Vec3(0,0,1), 10.0, float(nx)/float(ny), aperture, dist_to_focus, 0 ,1);
    }
}

__global__ void free_pool_table(Hittable **d_list, Hittable **d_world, Camera **d_camera) 
{
    for(int i=0; i < 17; i++) {
        delete ((Hittable *)d_list[i])->mat_ptr;
        delete d_list[i];
    }
    delete *d_world;
    delete *d_camera;
}

int main(int argc, char **argv)
{
    auto program_start = high_resolution_clock::now();

    /****** Set up image size, block size, and frame buffer ******/
    int depth = 50;
    int tx = 8;
    int ty = 8;

    int image, nx, ny, ns;
    if (argc < 5) {
        nx = 400;
        ny = 225;
        ns = 10000;
        image = 2;    
    } else {
        image = atoi(argv[1]);
        nx = atoi(argv[2]);
        ny = atoi(argv[3]);
        ns = atoi(argv[4]);
    }
    

    /****** Allocate and copy memory for any image textures ******/
    int tex_nx, tex_ny, tex_nn;
    int texHQ_nx, texHQ_ny, texHQ_nn;
    unsigned char *dev_mercury;
    unsigned char *dev_venus;
    unsigned char *dev_earth;
    unsigned char *dev_mars;
    unsigned char *dev_jupiter;
    unsigned char *dev_saturn;
    unsigned char *dev_uranus;
    unsigned char *dev_neptune;
    unsigned char *dev_sun;
    unsigned char *dev_pluto;

    auto texture_time_start = high_resolution_clock::now();

    if (image == 1) {
        /******  Standard quality textures ******/
        unsigned char *mercury = stbi_load("../Common/textures/mercury.jpg", &tex_nx, &tex_ny, &tex_nn, 0);
        unsigned char *venus = stbi_load("../Common/textures/venus.jpg", &tex_nx, &tex_ny, &tex_nn, 0);
        unsigned char *earth = stbi_load("../Common/textures/earth.jpg", &tex_nx, &tex_ny, &tex_nn, 0);
        unsigned char *mars = stbi_load("../Common/textures/mars.jpg", &tex_nx, &tex_ny, &tex_nn, 0);
        unsigned char *uranus = stbi_load("../Common/textures/uranus.jpg", &tex_nx, &tex_ny, &tex_nn, 0);
        unsigned char *neptune = stbi_load("../Common/textures/neptune.jpg", &tex_nx, &tex_ny, &tex_nn, 0);
        unsigned char *pluto = stbi_load("../Common/textures/pluto.jpg", &tex_nx, &tex_ny, &tex_nn, 0);

        /****** High quality textures for larger bodies ******/
        unsigned char *sun = stbi_load("../Common/textures/sunHQ.jpg", &texHQ_nx, &texHQ_ny, &texHQ_nn, 0);
        unsigned char *jupiter = stbi_load("../Common/textures/jupiterHQ.jpg", &texHQ_nx, &texHQ_ny, &texHQ_nn, 0);
        unsigned char *saturn = stbi_load("../Common/textures/saturnHQ.jpg", &texHQ_nx, &texHQ_ny, &texHQ_nn, 0);


        /****** Allocate memory and copy each texture to the GPU ******/
        size_t texSize = tex_nx*tex_ny*tex_nn*sizeof(unsigned char);
        size_t texHQSize = texHQ_nx*texHQ_ny*texHQ_nn*sizeof(unsigned char);
    
        checkCudaErrors(cudaMalloc((void **)&dev_mercury, texSize));
        checkCudaErrors(cudaMemcpy(dev_mercury, mercury, texSize, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMalloc((void **)&dev_venus, texSize));
        checkCudaErrors(cudaMemcpy(dev_venus, venus, texSize, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMalloc((void **)&dev_earth, texSize));
        checkCudaErrors(cudaMemcpy(dev_earth, earth, texSize, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMalloc((void **)&dev_mars, texSize));
        checkCudaErrors(cudaMemcpy(dev_mars, mars, texSize, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMalloc((void **)&dev_uranus, texSize));
        checkCudaErrors(cudaMemcpy(dev_uranus, uranus, texSize, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMalloc((void **)&dev_neptune, texSize));
        checkCudaErrors(cudaMemcpy(dev_neptune, neptune, texSize, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMalloc((void **)&dev_pluto, texSize));
        checkCudaErrors(cudaMemcpy(dev_pluto, pluto, texSize, cudaMemcpyHostToDevice));

        checkCudaErrors(cudaMalloc((void **)&dev_sun, texHQSize));
        checkCudaErrors(cudaMemcpy(dev_sun, sun, texHQSize, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMalloc((void **)&dev_jupiter, texHQSize));
        checkCudaErrors(cudaMemcpy(dev_jupiter, jupiter, texHQSize, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMalloc((void **)&dev_saturn, texHQSize));
        checkCudaErrors(cudaMemcpy(dev_saturn, saturn, texHQSize, cudaMemcpyHostToDevice));
    }

    /****** Allocate and copy memory for adjustable background color ******/
    Color background;
    if (image == 0) background = Color(0, 0, 0);
    else if (image == 1) background = Color(0, 0, 0);
    else if (image == 2) background = Color(0, 0, 0);
    else background = Color(0.70, 0.80, 1.00);

    Color *dev_background;
    checkCudaErrors(cudaMallocManaged((void **)&dev_background, sizeof(Color)));
    checkCudaErrors(cudaMemcpy(dev_background, &background, sizeof(Color), cudaMemcpyHostToDevice));

    auto texture_time_end = high_resolution_clock::now();


    std::cerr << "Rendering a " << nx << "x" << ny << " image with " << ns << " samples per pixel ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    auto create_time_start = high_resolution_clock::now();

    int num_pixels = nx*ny;
    size_t fb_size = num_pixels*sizeof(Vec3);

    // allocate frame buffer (unified memory)
    Vec3 *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    // allocate random state
    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels*sizeof(curandState)));

    curandState *d_rand_state2;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state2, 1*sizeof(curandState)));

    /****** Render and time frame buffer ******/
    
    // we need that 2nd random state to be initialized for the world creation
    rand_init<<<1,1>>>(d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // make our world of hittables
    Hittable **d_list;
    int numHittables;

    if (image == 0) {
        numHittables = 16 * 16 + 2;
        checkCudaErrors(cudaMalloc((void **)&d_list, numHittables*sizeof(Hittable *)));
    } else if (image == 1) {
        numHittables =  11+45*56;
        checkCudaErrors(cudaMalloc((void **)&d_list, numHittables*sizeof(Hittable *))); 
    } else if (image == 2) {
        numHittables = 18;
        checkCudaErrors(cudaMalloc((void **)&d_list, numHittables*sizeof(Hittable *))); 
    } else {
        numHittables = 22*22+1+4;
        checkCudaErrors(cudaMalloc((void **)&d_list, numHittables*sizeof(Hittable *))); 
    }

    Hittable **d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(Hittable *)));

    Camera **d_camera;
    checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(Camera *)));

    if (image == 0) {
        glow_balls<<<1,1>>>(d_list, d_world, d_camera, nx, ny, d_rand_state2);
    } else if (image == 1) {
        solar_system<<<1,1>>>(d_list,d_world,d_camera, nx, ny, d_rand_state2, tex_nx, tex_ny, texHQ_nx, texHQ_ny, dev_sun, 
                                                 dev_mercury, dev_venus, dev_earth, dev_mars, dev_jupiter, dev_saturn, dev_uranus, dev_neptune, dev_pluto);
    } else if (image == 2) {
        pool_table<<<1,1>>>(d_list, d_world, d_camera, nx, ny, d_rand_state2);
    } else {
        create_world<<<1,1>>>(d_list,d_world,d_camera, nx, ny, d_rand_state2);
    }

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Render our buffer
    dim3 blocks(nx/tx+1,ny/ty+1);
    dim3 threads(tx,ty);

    render_init<<<blocks, threads>>>(nx, ny, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    auto create_time_end = high_resolution_clock::now();
    std::cerr << "Starting Render.\n";

    auto render_time_start = high_resolution_clock::now();
    render<<<blocks, threads>>>(fb, nx, ny,  ns, d_camera, d_world, d_rand_state, depth, dev_background);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    
     std::cerr << "Render Finished.\n";

     auto render_time_end = high_resolution_clock::now();


    // Output File
    std::fstream file;
    file.open("out.ppm", std::ios::out);
   // std::streambuf *ppm_out = file.rdbuf();

     // Redirect Cout
   // std::cout.rdbuf(ppm_out);

    auto save_time_start = high_resolution_clock::now();

    // Output FB as Image
    file << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny-1;  j >= 0;  j--) {
        for (int i = 0;  i < nx;  i++) {
           size_t pixel_index = j*nx + i;
            writeColor(file,fb[pixel_index]);
        }
    }

    auto save_time_end = high_resolution_clock::now();

    // clean up
    checkCudaErrors(cudaDeviceSynchronize());
    if (image == 0) {
        free_glow_balls<<<1,1>>>(d_list, d_world, d_camera);
    } else if (image == 1) {
        free_solar_system<<<1,1>>>(d_list, d_world, d_camera);
        checkCudaErrors(cudaFree(dev_mercury));
        checkCudaErrors(cudaFree(dev_venus));
        checkCudaErrors(cudaFree(dev_earth));
        checkCudaErrors(cudaFree(dev_mars));
        checkCudaErrors(cudaFree(dev_jupiter));
        checkCudaErrors(cudaFree(dev_saturn));
        checkCudaErrors(cudaFree(dev_uranus));
        checkCudaErrors(cudaFree(dev_neptune));
        checkCudaErrors(cudaFree(dev_pluto));
        checkCudaErrors(cudaFree(dev_sun));
    } else if (image == 2) {
        free_pool_table<<<1,1>>>(d_list, d_world, d_camera);
    }else {
        free_world<<<1,1>>>(d_list, d_world, d_camera);
    }

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(d_rand_state2));
    checkCudaErrors(cudaFree(fb));
    checkCudaErrors(cudaFree(dev_background));

    std::cerr << "Image Successfully Saved." << std::endl;
    file.close();

    // useful for cuda-memcheck --leak-check full
    cudaDeviceReset();
    auto program_end = high_resolution_clock::now();

    // Texture Time 
    auto texture_time = duration_cast<milliseconds>(texture_time_end - texture_time_start);
    std::cout << "Texture Transfer Time: " << texture_time.count() << "ms" << std::endl;

    // Create Time 
    auto create_time = duration_cast<milliseconds>(create_time_end - create_time_start);
    std::cout << "World Creation Time: " << create_time.count() << "ms" << std::endl;

    // Render Time
    auto render_time = duration_cast<milliseconds>(render_time_end - render_time_start);
    std::cout << "Render Time: " << render_time.count() << "ms" << std::endl;

    // Save image time
    auto save_time = duration_cast<milliseconds>(save_time_end - save_time_start);
    std::cout << "Image Save Time: " << save_time.count() << "ms" << std::endl;
    
     // Total Time
    auto time = duration_cast<milliseconds>(program_end - program_start);
    std::cout << "Total Time: " << time.count() << "ms" << std::endl;
    return 0;
}