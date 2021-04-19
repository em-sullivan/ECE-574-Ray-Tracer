#ifndef PERLIN_H
#define PERLIN_H

#include "Vec3.h"

__device__  inline static Vec3 randomVec(float min, float max, curandState *local_rand_state) {
       float rand1 = curand_uniform(local_rand_state);
       rand1 *= (max -min+0.999999f);
       rand1 += min;
       rand1 = truncf(rand1);

        float rand2 = curand_uniform(local_rand_state);
       rand2 *= (max -min+0.999999f);
       rand2 += min;
       rand2 = truncf(rand2);

        float rand3 = curand_uniform(local_rand_state);
       rand3 *= (max -min+0.999999f);
       rand3 += min;
       rand3 = truncf(rand3);
       
        return Vec3(rand1, rand2, rand3);
    }


class Perlin {
    public:
        __device__ Perlin() {};
        __device__ Perlin(curandState *local_rand_state) {
            ranvec = new Vec3[point_count];
            for (int i = 0; i < point_count; ++i) {
                ranvec[i] = unitVector(randomVec(-1,1, local_rand_state));
            }

            perm_x = perlin_generate_perm(local_rand_state);
            perm_y = perlin_generate_perm(local_rand_state);
            perm_z = perlin_generate_perm(local_rand_state);
        }

        __device__ ~Perlin() {
            delete[] ranvec;
            delete[] perm_x;
            delete[] perm_y;
            delete[] perm_z;
        }

        __device__ float noise(const Point3& p) const {
            float u = p.x() - floorf(p.x());
            float v = p.y() - floorf(p.y());
            float w = p.z() - floorf(p.z());
            int i = static_cast<int>(floorf(p.x()));
            int j = static_cast<int>(floorf(p.y()));
            int k = static_cast<int>(floorf(p.z()));
            Vec3 c[2][2][2];

            for (int di=0; di < 2; di++)
                for (int dj=0; dj < 2; dj++)
                    for (int dk=0; dk < 2; dk++)
                        c[di][dj][dk] = ranvec[
                            perm_x[(i+di) & 255] ^
                            perm_y[(j+dj) & 255] ^
                            perm_z[(k+dk) & 255]
                        ];

            return perlin_interp(c, u, v, w);
        }

        __device__ float turb(const Point3& p, int depth=7) const {
            float accum = 0.0f;
            auto temp_p = p;
            float weight = 1.0f;

            for (int i = 0; i < depth; i++) {
                accum += weight * noise(temp_p);
                weight *= 0.5f;
                temp_p *= 2;
            }

            return fabsf(accum);
        }

    private:
        static const int point_count = 256;
        Vec3* ranvec;
        int* perm_x;
        int* perm_y;
        int* perm_z;

        __device__ static int* perlin_generate_perm(curandState *local_rand_state) {
            auto p = new int[point_count];

            for (int i = 0; i < point_count; i++)
                p[i] = i;

            permute(p, point_count, local_rand_state);

            return p;
        }

        __device__ static void permute(int* p, int n, curandState *local_rand_state) {
            for (int i = n-1; i > 0; i--) {
                float rand1 = curand_uniform(local_rand_state);
                rand1 *= (i+0.999999f);
                int target = (int)truncf(rand1);
                int tmp = p[i];
                p[i] = p[target];
                p[target] = tmp;
            }
        }

        __device__ static float perlin_interp(Vec3 c[2][2][2], float u, float v, float w) {
            float uu = u*u*(3.f-2.f*u);
            float vv = v*v*(3.f-2.f*v);
            float ww = w*w*(3.f-2.f*w);
            float accum = 0.0f;

            for (int i=0; i < 2; i++)
                for (int j=0; j < 2; j++)
                    for (int k=0; k < 2; k++) {
                        Vec3 weight_v(u-i, v-j, w-k);
                        accum += (i*uu + (1.f-i)*(1.f-uu))*
                            (j*vv + (1.f-j)*(1.f-vv))*
                            (k*ww + (1.f-k)*(1.f-ww))*dot(c[i][j][k], weight_v);
                    }

            return accum;
        }
};


#endif