/*
 * Perlin Noise Class
 * This is used for creating solid looking
 * mapped textures
 */

#ifndef PERLIN_H
#define PERLIN_H

#include "Vec3.h"
#include "Ray.h"
#include "shader_consts.h"

class Perlin
{
public:
    // Constructor
    __device__ Perlin() {};
    __device__ Perlin(curandState *randState);

    // Deconstructor
    __device__  ~Perlin();

    __device__  float noise(const Point3 &p) const;
    __device__  float turb(const Point3 &p, int depth = 7) const;

private:
    static const int point_count = 256;
    Vec3 *ranvec;
    int *perm_x;
    int *perm_y;
    int *perm_z;

    __device__ static int* perlinGeneratePerm(curandState *randState);
    __device__  static void permute(int *p, int n, curandState *randState);
    __device__  static float perlinInterp(Vec3 c[2][2][2], float u, float v, float w);
};

#endif // PERLIN_H