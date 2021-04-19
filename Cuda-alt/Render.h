#ifndef RENDER_H
#define RENDER_H

#include <curand_kernel.h>

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

__device__ Color ray_color(const Ray& r, Hittable **world, curandState *local_rand_state, int depth, Color* background) 
{
    Ray cur_ray = r;
    Vec3 cur_attenuation = Color(1.0, 1.0, 1.0);
    Color emitted;

    for (int i = 0; i < depth; i++) {
        hit_record rec;

        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
            Ray scattered;
            Vec3 attenuation;
            emitted = rec.mat_ptr->emitted(rec.u,rec.v, rec.p);
            if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            } else return emitted*cur_attenuation;
        } else {
           // background
           return emitted + cur_attenuation *(*background);
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
    //curand_init(1984, pixel_index, 0,  &rand_state[pixel_index]);
    curand_init(1984+pixel_index, 0, 0, &rand_state[pixel_index]);
}

__global__ void render(Vec3 *fb, int max_x, int max_y, int ns, Camera **cam, Hittable **world, curandState *rand_state, int depth, Color* background)
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
        pixel_color += ray_color(r, world, &local_rand_state, depth, background);
    }
    rand_state[pixel_index] = local_rand_state;
    pixel_color /= float(ns);
    pixel_color[0] = sqrt(pixel_color[0]);
    pixel_color[1] = sqrt(pixel_color[1]);
    pixel_color[2] = sqrt(pixel_color[2]);
    fb[pixel_index] = pixel_color;
}

#endif