#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
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
//#include "Material.h"

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line)
{
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__device__ Color ray_color(const Ray& r, Hittable **world, curandState *local_rand_state, int depth) 
{
    Ray cur_ray = r;
    Vec3 cur_attenuation = Color(1.0, 1.0, 1.0);

    for (int i = 0; i < depth; i++) {
        hit_record rec;

        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
            Ray scattered;
            Vec3 attenuation;
            if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            } else return Vec3(0.0, 0.0, 0.0);
        } else {
            Vec3 unit_direction = unitVector(cur_ray.direction());
            float t = 0.5f * (unit_direction.y() + 1.0f);
            Vec3 c = (1.0f - t) * Vec3(1.0, 1.0, 1.0) + t * Vec3(0.5, 0.7, 1.0);
            return cur_attenuation * c;
        }
    }
    return Vec3(0.0, 0.0, 0.0); // exceeded recursion
}

__global__ void rand_init(curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(1984, 0, 0, rand_state);
    }
}


__global__ void render_init(int max_x, int max_y, curandState *rand_state)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    //Each thread gets same seed, a different sequence number, no offset
    curand_init(1984, pixel_index, 0,  &rand_state[pixel_index]);
   // curand_init(1984+pixel_index, 0, 0, &rand_state[pixel_index]);
}

__global__ void render(Vec3 *fb, int max_x, int max_y, int ns, Camera **cam, Hittable **world, curandState *rand_state, int depth)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    
    int pixel_index = j*max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    Color pixel_color(0,0,0);
    
    for(int s=0; s < ns; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        Ray r = (*cam)->get_ray(u,v, &local_rand_state);
        pixel_color += ray_color(r, world, &local_rand_state, depth);
    }
    rand_state[pixel_index] = local_rand_state;
    pixel_color /= float(ns);
    pixel_color[0] = sqrt(pixel_color[0]);
    pixel_color[1] = sqrt(pixel_color[1]);
    pixel_color[2] = sqrt(pixel_color[2]);
    fb[pixel_index] = pixel_color;
}
/*
__global__ void create_world(Hittable **d_list, Hittable **d_world, Camera **d_camera, int nx, int ny)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_list[0] = new Sphere(Vec3(0,0,-1), 0.5, new Lambertian(new Solid_Color(Vec3(0.1, 0.2, 0.5))));
        d_list[1] = new Sphere(Vec3(0,-100.5,-1), 100, new Lambertian(new Solid_Color(Vec3(0.8, 0.8, 0.0))));
        d_list[2] = new Sphere(Vec3(1,0,-1), 0.5, new Metal(Vec3(0.8, 0.6, 0.2), 0.0));
        d_list[3] = new Sphere(Vec3(-1,0,-1), 0.5, new Dielectric(1.5));
        d_list[4] = new Sphere(Vec3(-1,0,-1), -0.45, new Dielectric(1.5));
        *d_world = new Hittable_List(d_list,5);
        *d_camera   = new Camera(Point3(-2,2,1),Point3(0,0,-1),Vec3(0,1,0), 20.0f, float(nx)/float(ny), 0, 10.0, 0.0, 1.0);
    }
}
*/

#define RND (curand_uniform(&local_rand_state))

__global__ void create_world(Hittable **d_list, Hittable **d_world, Camera **d_camera, int nx, int ny, curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;
        d_list[0] = new Sphere(Vec3(0,-1000.0,-1), 1000, new Lambertian(new Solid_Color(Vec3(0.5, 0.5, 0.5))));
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
        d_list[i++] = new Sphere(Vec3(-4, 1, 0), 1.0, new Lambertian(new Solid_Color(Vec3(0.4, 0.2, 0.1))));
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

__global__ void free_world(Hittable **d_list, Hittable **d_world, Camera **d_camera) {
    for(int i=0; i < 22*22+1+3; i++) {
        delete ((Sphere *)d_list[i])->mat_ptr;
        delete d_list[i];
    }
    delete *d_world;
    delete *d_camera;
}

int main(int argc, char **argv)
{
    /****** Set up image size, block size, and frame buffer ******/
    int nx = 800;
    int ny = 800;
    int depth = 50;
    int ns = 20;
    int tx = 8;
    int ty = 8;
    

    std::cerr << "Rendering a " << nx << "x" << ny << " image with " << ns << " samples per pixel ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    int num_pixels = nx*ny;
    size_t fb_size = num_pixels*sizeof(Vec3);

    // allocate frame buffer (unified memory)
    Vec3 *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

      std::cerr << "Before rand.\n";

    // allocate random state
    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels*sizeof(curandState)));

    curandState *d_rand_state2;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state2, 1*sizeof(curandState)));

     // we need that 2nd random state to be initialized for the world creation
    rand_init<<<1,1>>>(d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

      std::cerr << "Before list.\n";

    // make our world of hittables
    Hittable **d_list;
    int numHittables = 22*22+1+3;
    checkCudaErrors(cudaMalloc((void **)&d_list, numHittables*sizeof(Hittable *)));

    std::cerr << "Before world.\n";
    Hittable **d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(Hittable *)));

    std::cerr << "Before camera.\n";
    Camera **d_camera;
    checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(Camera *)));

    std::cerr << "Before create world.\n";
    create_world<<<1,1>>>(d_list,d_world,d_camera, nx, ny, d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    /****** Render and time frame buffer ******/
    clock_t start, stop;
    start = clock();

    // Render our buffer
    dim3 blocks(nx/tx+1,ny/ty+1);
    dim3 threads(tx,ty);
    std::cerr << "Before Render Init.\n";

    render_init<<<blocks, threads>>>(nx, ny, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

      std::cerr << "Before Render.\n";

    render<<<blocks, threads>>>(fb, nx, ny,  ns, d_camera, d_world, d_rand_state, depth);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    
     std::cerr << "After Render.\n";

    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cout << "took " << timer_seconds << " seconds.\n";


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

    std::cerr << "\nDone" << std::endl;
    file.close();

    // useful for cuda-memcheck --leak-check full
    cudaDeviceReset();
    
    return 0;
}