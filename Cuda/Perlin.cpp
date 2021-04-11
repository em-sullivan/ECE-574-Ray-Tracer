/*
 * Source file for Perlin.
 * Used for generating Perlin noise
 */

#include "Perlin.h"

__device__  Perlin::Perlin(curandState *randState)
{
    // Generate an array of random floats
    ranvec = new Vec3[point_count];
    for (int i = 0; i < point_count; i++) {
        ranvec[i] = unitVector(Vec3::random(-1, 1, randState));
    }

    // Generate perms
    perm_x = perlinGeneratePerm(randState);
    perm_y = perlinGeneratePerm(randState);
    perm_z = perlinGeneratePerm(randState);
}

__device__  Perlin::~Perlin()
{
    // Free allocated memory
    delete[] ranvec;
    delete[] perm_x;
    delete[] perm_y;
    delete[] perm_z;
}

__device__  float Perlin::noise(const Point3 &p) const
{
    auto u = p.x() - floorf(p.x());
    auto v = p.y() - floorf(p.y());
    auto w = p.z() - floorf(p.z());

    // Hermitian smoothing
    u = u * u * (3 - 2 * u);
    v = v * v * (3 - 2 * v);
    w = w * w * (3 - 2 * w);   

    auto i = static_cast<int>(floor(p.x()));
    auto j = static_cast<int>(floor(p.y()));
    auto k = static_cast<int>(floor(p.z()));
    Vec3 c[2][2][2];

    for (int di = 0; di < 2; di++) {
        for (int dj = 0; dj < 2; dj++) {
            for(int dk = 0; dk < 2; dk++) {
                c[di][dj][dk] = ranvec[perm_x[(i + di) & 255] ^
                    perm_y[(j + dj) & 255] ^
                    perm_z[(k + dk) & 255]];
            }
        }
    }

    return perlinInterp(c, u, v, w);
}

__device__  float Perlin::turb(const Point3 &p, int depth) const
{
    auto accum = 0.0;
    auto temp_p = p;
    auto weight = 1.0;

    for (int i = 0; i < depth; i++) {
        accum += weight * noise(temp_p);
        weight *= 0.5;
        temp_p *= 2;
    }

    return fabs(accum);
}

__device__  int* Perlin::perlinGeneratePerm(curandState *randState)
{
    auto p = new int[point_count];

    for (int i = 0; i < Perlin::point_count; i++)
        p[i] = i;

    permute(p, point_count, randState);
    return p;
}

__device__  void Perlin::permute(int *p, int n, curandState *randState)
{
    int tmp;

    for (int i = n - 1; i > 0; i--) {
        int target = random_int(0, i, randState);
        tmp = p[i];
        p[i] = p[target];
        p[target] = tmp;
    }
}

__device__  float Perlin::perlinInterp(Vec3 c[2][2][2], float u, float v, float w)
{
    auto uu = u * u * (3 - 2 * u);
    auto vv = v * v * (3 - 2 * v);
    auto ww = w * w * (3 - 2 * w);
    float accum = 0.0;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
                Vec3 weight_v(u - i, v - j, w - k);
                accum += (i * uu + (1 - i) * (1-uu)) *
                    (j * vv + (1 - j)*(1 - vv)) *
                    (k * ww + (1 - k)*(1 - ww)) * 
                    dot(c[i][j][k], weight_v);
            }
        }
    }

    return accum;
}