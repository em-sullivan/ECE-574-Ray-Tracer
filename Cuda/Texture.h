/*
 * Texture Class
 */

#ifndef TEXTURE_H
#define TEXTURE_H

#include <iostream>
#include "Vec3.h"
#include "Ray.h"
#include "Perlin.h"
#include "shader_consts.h"
//#include "shader_stb_image.h"
//#define STB_IMAGE_IMPLEMENTATION
//#include "stb_image.h"


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
 * Noise - Perlin noise texture
 */

class Noise_Text : public Texture
{
public:
    // Constructor
    __device__  Noise_Text();
    __device__  Noise_Text(float sc);

    // Color value
    __device__  virtual Color value(float u, float v, const Point3 &p) const override;

private:
    Perlin noise;
    float scale;

};

/*
 * Image texture
 */

class Image_Text : public Texture
{
public: 
    const static int bytes_per_pixel = 3;

    __device__ Image_Text();
    __device__ Image_Text(const char *filename);
    __device__  ~Image_Text();

    __device__  virtual Color value(float u, float v, const Point3 &p) const override;

private:
    unsigned char *data;
    int width, height;
    int bytes_per_scaneline;

};

#endif //TEXTURE_H