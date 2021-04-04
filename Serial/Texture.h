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
    virtual Color value(float u, float v, const Point3 &p) const = 0;
};

/*
 * Texture: Solid colors - Just give it an RGB value with Color
 */
class Solid_Color : public Texture
{
public:

    // Constructors
    Solid_Color();
    Solid_Color(Color c);
    Solid_Color(float r, float g, float b);

    virtual Color value(float u, float v, const Point3 &p) const override;

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
    Checkered();
    Checkered(shared_ptr<Texture> even, shared_ptr<Texture> odd);
    Checkered(Color c1, Color c2);

    // Color value
    virtual Color value(float u, float v, const Point3 &p) const override;


private:
    // Textures for each tile
    shared_ptr<Texture> odd_tiles;
    shared_ptr<Texture> even_tiles;

};

/*
 * Noise - Perlin noise texture
 */

class Noise_Text : public Texture
{
public:
    // Constructor
    Noise_Text();
    Noise_Text(float sc);

    // Color value
    virtual Color value(float u, float v, const Point3 &p) const override;

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

    Image_Text();
    Image_Text(const char *filename);
    ~Image_Text();

    virtual Color value(float u, float v, const Point3 &p) const override;

private:
    unsigned char *data;
    int width, height;
    int bytes_per_scaneline;

};

#endif //TEXTURE_H