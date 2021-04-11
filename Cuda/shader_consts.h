/*
 * Constants and declaraitons
 * for this simple shader program
 */

#ifndef SHADER_CONSTS_H
#define SHADER_CONSTS_H

#include <limits>
#include <memory>
#include <cstdlib>
#include<curand_kernel.h>

// Constants
//#define INF std::numeric_limits<float>::infinity()
#define PI 3.1415926535897932385

// Utility function to convert degrees to rads
__device__ inline float deg_to_rad(float degrees)
{
    return degrees * PI / 180.0;
}

// Issue with rand() being on the CPU but needed on device, fix later
__device__ inline float random_float(curandState *rand_state)
{
    // Return a (pseduo) random float between 0.0 and 1.0
    return curand_uniform(rand_state) / (RAND_MAX + 1.0);
}

__device__ inline float random_float(float min, float max, curandState *rand_state)
{
    // Return a (pseduo) random float between the min and max value
    return min + (max - min) * random_float(rand_state);
}


__host__ __device__ inline float clamp(float x, float min, float max)
{
    if (x < min) return min;
    if (x > max) return max;
    return x;
}

__device__ inline int random_int(int min, int max, curandState *randState)
{
    return static_cast<int>(random_float((float)min, (float)(max + 1), randState));
}

// Some common headers
//#include "Ray.h"
//#include "Vec3.h"

#endif // SHADER_CONSTS_H