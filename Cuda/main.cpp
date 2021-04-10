#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <time.h>
#include "Vec3.h"
#include "Color.h"
#include "Ray.h"

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

__device__ Color  ray_color(const Ray& r) 
{
   Vec3 unit_direction = unitVector(r.direction());
   float t = 0.5f*(unit_direction.y() + 1.0f);
   return (1.0f-t)*Vec3(1.0, 1.0, 1.0) + t*Vec3(0.5, 0.7, 1.0);
}

__global__ void render(Vec3 *fb, int max_x, int max_y, Vec3 lower_left_corner, Vec3 horizontal, Vec3 vertical, Vec3 origin)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if((i >= max_x) || (j >= max_y)) return;
 
    int pixel_index = j*max_x + i;

    float u = float(i) / float(max_x);
    float v = float(j) / float(max_y);
    Ray r(origin, lower_left_corner + u*horizontal + v*vertical);
    
    fb[pixel_index] = ray_color(r);
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
                                Vec3(0.0, 0.0, 0.0));
                    
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
    for (int j = ny-1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
           size_t pixel_index = j*nx + i;
            int ir = int(255.99*fb[pixel_index].x());
            int ig = int(255.99*fb[pixel_index].y());
            int ib = int(255.99*fb[pixel_index].z());
            std::cout << ir << " " << ig << " " << ib << "\n";
            //writeColor(std::cout,fb[pixel_index],1);
        }
    }

    checkCudaErrors(cudaFree(fb));

    std::cerr << "\nDone" << std::endl;
    file.close();
    return 0;
}