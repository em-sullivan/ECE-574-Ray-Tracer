/*
 * Texture Class
 */

#ifndef TEXTURE_H
#define TEXTURE_H

#include <iostream>
#include "Vec3.h"
#include "Ray.h"
#include "shader_consts.h"

class Texture
{
public:
    __device__ virtual Color value(float u, float v, const Point3 &p) const = 0;
};

/*
 * Texture: Solid colors - Just give it an RGB value with Color
 */
class Solid_Color : public Texture
{
public:

    // Constructors
    __device__  Solid_Color();
    __device__  Solid_Color(Color c);
    __device__  Solid_Color(float r, float g, float b);

    __device__  virtual Color value(float u, float v, const Point3 &p) const override;

private:
    Color color_v;

};

/*
 * Checked - two Solid_Color textures - neato
 */
class Checkered : public Texture
{
public:
    // Constructors
    __device__  Checkered();
    __device__  Checkered(Texture *even, Texture *odd);
    __device__  Checkered(Color c1, Color c2);

    // Color value
    __device__  virtual Color value(float u, float v, const Point3 &p) const override;


private:
    // Textures for each tile
    Texture *odd_tiles;
    Texture *even_tiles;

};

/*
 * Solid Color
 */
__device__  Solid_Color::Solid_Color(Color c)
{
    color_v = c;
}

__device__  Solid_Color::Solid_Color(float r, float g, float b)
{
    color_v = Color(r, g, b);
}

__device__  Color Solid_Color::value(float u, float v, const Point3 &p) const
{
    return color_v;
}

/*
 * Solid Checkard pattern
 */

__device__  Checkered::Checkered(Texture *even, Texture *odd)
{
    even_tiles = even;
    odd_tiles = odd;
}

// Need to fix memory allocation on CUDA
__device__  Checkered::Checkered(Color c1, Color c2)
{
    even_tiles = new Solid_Color(c1);
    odd_tiles = new Solid_Color(c2);
}

__device__  Color Checkered::value(float u, float v, const Point3 &p) const
{
    float sines = sinf(10.f * p.x()) * sinf(10.f * p.y()) * sinf(10.f * p.z());
    if (sines < 0.f)
        return odd_tiles->value(u, v, p);
    else
        return even_tiles->value(u, v, p);
}

/*
class Noise_Text : public Texture
{
public:
    // Constructor
    __device__  Noise_Text()
    {
        scale = 1.0f;
    }
    __device__  Noise_Text(float sc)
    {
        scale = sc;
    }
    // Color value
    __device__  virtual Color value(float u, float v, const Point3 &p) const override
    {
    // Generator noise
    // return Color(1, 1, 1) * noise.turb(p * scale);
    // Comments out straight noise, igves a marbel like effect
        return Color(1, 1, 1) * 0.5f * (1.f + sinf(scale * p.z() + 10.f * noise.turb(p)));
    }

private:
    Perlin noise;
    float scale;

};
*/
class Image_Text : public Texture
{
public: 
    const static int bytes_per_pixel = 3;

    __device__ Image_Text();
    __device__ Image_Text(unsigned char *pixels, int A, int B);

    __device__  ~Image_Text();

    __device__  virtual Color value(float u, float v, const Point3 &p) const override;

private:
    unsigned char *data;
    int width, height;
    int bytes_per_scaneline;
};

__device__ Image_Text::Image_Text(unsigned char *pixels, int A, int B)
{
    data = pixels;
    width = A;
    height = B;
    bytes_per_scaneline = bytes_per_pixel * width;
}

__device__  Image_Text::~Image_Text()
{
    delete data;
}

__device__ inline float clamp_t(float x, float min, float max)
{
    if (x < min) return min;
    if (x > max) return max;
    return x;
}

__device__  Color Image_Text::value(float u, float v, const Point3 &p) const
{
    // No text data, return cyan for debugging
    if (data == nullptr)
        return Color(0, 1, 1);

    // Clamp input text cordinate to [0, 1] x [1, 0]
    u = clamp_t(u, 0.0, 1);
    // Flip V to image coordinates
    v = 1.0f - clamp_t(v, 0.0, 1.0); 

    auto i = static_cast<int>(u * width);
    auto j = static_cast<int>(v * height);

    // Clamp integer mapping, actual coordinates should be less than 1.0
    if (i >= width) i = width - 1;
    if (j >= height) j = height - 1;

    const auto color_scale = 1.0f / 255.0f;
    auto pixel = data + j * bytes_per_scaneline + i * bytes_per_pixel;

    return Color(color_scale * pixel[0], color_scale * pixel[1], color_scale * pixel[2]);
}

/*
class image_texture : public Texture {
    public:
    __device__ image_texture() {}
    __device__ image_texture(unsigned char *pixels, int A, int B) : data(pixels), nx(A), ny(B) {}
    __device__ virtual Vec3 value(float u, float v, const Vec3& p) const;
        unsigned char *data;
        int nx, ny;
};



__device__ Vec3 image_texture::value(float u, float v, const Vec3& p) const {
        if (data == nullptr)
        return Color(1, 0, 1);

    
     int i = (  u)*nx;
     int j = (1-v)*ny-0.001f;
     if (i < 0) i = 0;
     if (j < 0) j = 0;
     if (i > nx-1) i = nx-1;
     if (j > ny-1) j = ny-1;
     float r = int(data[3*i + 3*nx*j]  ) / 255.0f;
     float g = int(data[3*i + 3*nx*j+1]) / 255.0f;
     float b = int(data[3*i + 3*nx*j+2]) / 255.0f;
     return Vec3(r, g, b);
}
*/


#endif //TEXTURE_H