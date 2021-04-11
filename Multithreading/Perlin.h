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
    Perlin();

    // Deconstructor
    ~Perlin();

    float noise(const Point3 &p) const;
    float turb(const Point3 &p, int depth = 7) const;

private:
    static const int point_count = 256;
    Vec3 *ranvec;
    int *perm_x;
    int *perm_y;
    int *perm_z;

    static int* perlinGeneratePerm();
    static void permute(int *p, int n);
    static float perlinInterp(Vec3 c[2][2][2], float u, float v, float w);
};

#endif // PERLIN_H