#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>
#include <curand_kernel.h>
#include "Vec3.h"
#include "Color.h"
#include "Ray.h"
#include "Sphere.h"
#include "Hittable_List.h"
#include "Camera.h"
#include "Texture.h"
#include "Render.h"
//#include "Material.h"

#include "shader_stb_image.h"

#define RND (curand_uniform(&local_rand_state))

__global__ void create_world(Hittable **d_list, Hittable **d_world, Camera **d_camera, int nx, int ny, curandState *rand_state, int tex_nx, int tex_ny, unsigned char *tex_data) 
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;
        Texture *checker = new Checkered( new Solid_Color(Vec3(0.2,0.3, 0.1)), new Solid_Color(Vec3(0.3, 0.1, 0.2)));
        d_list[0] = new Sphere(Vec3(0,-1000.0,-1), 1000, new Lambertian(checker));
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

        d_list[i++] = new Sphere(Vec3(-4, 1,0),  1.0, new Dielectric(1.5));
        d_list[i++] = new Sphere(Vec3(0, 1, 0), 1.0 , new Lambertian(new Image_Text(tex_data, tex_nx, tex_ny)));
        d_list[i++] =  new Sphere(Vec3(0, 4, 5),  1.0, new Diffuse_Light( new Solid_Color(Vec3(7, 7, 7))));
        d_list[i++] = new Sphere(Vec3(4, 1, 0),  1.0, new Metal(Vec3(0.7, 0.6, 0.5), 0.0));

        
        *rand_state = local_rand_state;
        *d_world  = new Hittable_List(d_list, 22*22+1+4);

        Vec3 lookfrom = Vec3(13,2,3);
        Vec3 lookat = Vec3(0,0,0);
        float dist_to_focus = 10.0;
        float aperture = 0.1;
        *d_camera   = new Camera(lookfrom, lookat, Vec3(0,1,0), 25.0, float(nx)/float(ny), aperture, dist_to_focus, 0 ,1);
    }
}

__global__ void free_world(Hittable **d_list, Hittable **d_world, Camera **d_camera) 
{
    for(int i=0; i < 22*22+1+4; i++) {
        delete ((Sphere *)d_list[i])->mat_ptr;
        delete d_list[i];
    }
    delete *d_world;
    delete *d_camera;
}

int main(int argc, char **argv)
{
    /****** Set up image size, block size, and frame buffer ******/
    int nx = 1200;
    int ny = 800;
    int depth = 50;
    int ns = 100;
    int tx = 8;
    int ty = 8;

    /****** Allocate and copy memory for any image textures ******/
    int tex_nx, tex_ny, tex_nn;
    unsigned char *tex_data = stbi_load("textures/earthmap.jpg", &tex_nx, &tex_ny, &tex_nn, 0);
    unsigned char *dev_tex_data;

    checkCudaErrors(cudaMalloc((void **)&dev_tex_data, tex_nx*tex_ny*tex_nn*sizeof(unsigned char)));
    checkCudaErrors(cudaMemcpy(dev_tex_data, tex_data, tex_nx *tex_ny*tex_nn*sizeof(unsigned char), cudaMemcpyHostToDevice));

    /****** Allocate and copy memory for adjustable background color ******/
    Color background = Color(0, 0, 0);
    Color *dev_background;
    checkCudaErrors(cudaMallocManaged((void **)&dev_background, sizeof(Color)));
    checkCudaErrors(cudaMemcpy(dev_background, &background, sizeof(Color), cudaMemcpyHostToDevice));


    std::cerr << "Rendering a " << nx << "x" << ny << " image with " << ns << " samples per pixel ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

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

     // we need that 2nd random state to be initialized for the world creation
    rand_init<<<1,1>>>(d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // make our world of hittables
    Hittable **d_list;
    int numHittables = 22*22+1+4;
    checkCudaErrors(cudaMalloc((void **)&d_list, numHittables*sizeof(Hittable *)));

    Hittable **d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(Hittable *)));

    Camera **d_camera;
    checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(Camera *)));

    create_world<<<1,1>>>(d_list,d_world,d_camera, nx, ny, d_rand_state2, tex_nx, tex_ny, dev_tex_data);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    /****** Render and time frame buffer ******/
    clock_t start, stop;
    start = clock();

    // Render our buffer
    dim3 blocks(nx/tx+1,ny/ty+1);
    dim3 threads(tx,ty);

    render_init<<<blocks, threads>>>(nx, ny, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

      std::cerr << "Starting Render.\n";

    render<<<blocks, threads>>>(fb, nx, ny,  ns, d_camera, d_world, d_rand_state, depth, dev_background);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    
     std::cerr << "Render Finished.\n";

    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cout << "Render took " << timer_seconds << " seconds.\n";


    // Output File
    std::fstream file;
    file.open("out.ppm", std::ios::out);
    std::streambuf *ppm_out = file.rdbuf();

     // Redirect Cout
    std::cout.rdbuf(ppm_out);

    // Output FB as Image
    std::cout << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny-1;  j >= 0;  j--) {
        for (int i = 0;  i < nx;  i++) {
           size_t pixel_index = j*nx + i;
            writeColor(std::cout,fb[pixel_index]);
        }
    }

    // clean up
    checkCudaErrors(cudaDeviceSynchronize());
    free_world<<<1,1>>>(d_list,d_world,d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(d_rand_state2));
    checkCudaErrors(cudaFree(fb));
    checkCudaErrors(cudaFree(dev_tex_data));
    checkCudaErrors(cudaFree(dev_background));

    std::cerr << "Done" << std::endl;
    file.close();

    // useful for cuda-memcheck --leak-check full
    cudaDeviceReset();
    
    return 0;
}