#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <time.h>
#include <float.h>
#include "Vec3.h"
#include "Color.h"
#include "Ray.h"
#include "Sphere.h"
#include "Hittable_list.h"

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

__device__ Color  ray_color(const Ray& r, Hittable **world) 
{
    hit_record rec;
    if ((*world)->hit(r, 0.0, INF, rec)) {
        return 0.5f*Vec3(rec.normal.x()+1.0f, rec.normal.y()+1.0f, rec.normal.z()+1.0f);
    }
    else {
        Vec3 unit_direction = unitVector(r.direction());
        float t = 0.5f*(unit_direction.y() + 1.0f);
        return (1.0f-t)*Vec3(1.0, 1.0, 1.0) + t*Vec3(0.5, 0.7, 1.0);
    }
}

__global__ void render(Vec3 *fb, int max_x, int max_y, Vec3 lower_left_corner, Vec3 horizontal, Vec3 vertical, Vec3 origin, Hittable **world)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if((i >= max_x) || (j >= max_y)) return;
 
    int pixel_index = j*max_x + i;

    float u = float(i) / float(max_x);
    float v = float(j) / float(max_y);
    Ray r(origin, lower_left_corner + u*horizontal + v*vertical);
    
    fb[pixel_index] = ray_color(r, world);
}

__global__ void create_world(Hittable **d_list, Hittable **d_world)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *(d_list)   = new Sphere(Vec3(0, 0, -1), 0.5);
        *(d_list+1) = new Sphere(Vec3(0, -100.5, -1), 100);
        *d_world    = new Hittable_List(d_list, 2);
    }
}

__global__ void free_world(Hittable **d_list, Hittable **d_world)
{
    delete *(d_list);
    delete *(d_list+1);
    delete *d_world;
}

int main(int argc, char **argv)
{
    /****** Set up image size, block size, and frame buffer ******/
    int nx = 1200;
    int ny = 600;
    int tx = 8;
    int ty = 8;

    std::cerr << "Rendering a " << nx << "x" << ny << " image ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    int num_pixels = nx*ny;
    size_t fb_size = 3*num_pixels*sizeof(float);

    // allocate frame buffer (unified memory)
    Vec3 *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    // make our world of hitables
    Hittable **d_list;
    checkCudaErrors(cudaMalloc((void **)&d_list, 2*sizeof(Hittable *)));
    Hittable **d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(Hittable *)));
    create_world<<<1,1>>>(d_list,d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    /****** Render and time frame buffer ******/
    clock_t start, stop;
    start = clock();

    // Render our buffer
    dim3 blocks(nx/tx+1,ny/ty+1);
    dim3 threads(tx,ty);

    render<<<blocks, threads>>>(fb, nx, ny,
                                Vec3(-2.0, -1.0, -1.0),
                                Vec3(4.0, 0.0, 0.0),
                                Vec3(0.0, 2.0, 0.0),
                                Vec3(0.0, 0.0, 0.0),
                                d_world);
                    
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";


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
            int ir = int(255.99*fb[pixel_index].x());
            int ig = int(255.99*fb[pixel_index].y());
            int ib = int(255.99*fb[pixel_index].z());
            std::cout << ir << " " << ig << " " << ib << "\n";
            //writeColor(std::cout,fb[pixel_index],1);
        }
    }

    // clean up
    checkCudaErrors(cudaDeviceSynchronize());
    free_world<<<1,1>>>(d_list, d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(fb));

    std::cerr << "\nDone" << std::endl;
    file.close();

    // useful for cuda-memcheck --leak-check full
    cudaDeviceReset();
    
    return 0;
}