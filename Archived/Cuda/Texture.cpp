/*
 * Texture class source file
 * Makes colors on surfaces procedural
 */

#include "Texture.h"

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
__device__  Noise_Text::Noise_Text()
{
    scale = 1.0;
}

__device__  Noise_Text::Noise_Text(float sc)
{
    scale = sc;
}

__device__  Color Noise_Text::value(float u, float v, const Point3 &p) const
{
    // Generator noise
    // return Color(1, 1, 1) * noise.turb(p * scale);
    // Comments out straight noise, igves a marbel like effect
    return Color(1, 1, 1) * 0.5 * (1 + sin(scale * p.z() + 10 * noise.turb(p)));
}

__host__  Image_Text::Image_Text()
{
    data = nullptr;
    width = 0;
    height = 0;
    bytes_per_scaneline = 0;
}

__host__ Image_Text::Image_Text(const char *filename)
{
    auto components_per_pixel = bytes_per_pixel;

    // Load texture data from image, stored in unsighend char array
    data = stbi_load(filename, &width, &height, 
        &components_per_pixel, components_per_pixel);

    if (!data) {
        std::cerr << "Error! Could not texture for image  " << filename << std::endl;
        width = 0;
        height = 0;
    }

    bytes_per_scaneline = bytes_per_pixel * width;
}

__host__  Image_Text::~Image_Text()
{
    delete data;
}

__device__  Color Image_Text::value(float u, float v, const Point3 &p) const
{
    // No text data, return cyan for debugging
    if (data == nullptr)
        return Color(0, 1, 1);

    // Clamp input text cordinate to [0, 1] x [1, 0]
    u = clamp(u, 0.0, 1);
    // Flip V to image coordinates
    v = 1.0 - clamp(v, 0.0, 1.0); 

    auto i = static_cast<int>(u * width);
    auto j = static_cast<int>(v * height);

    // Clamp integer mapping, actual coordinates should be less than 1.0
    if (i >= width) i = width - 1;
    if (j >= height) j = height - 1;

    const auto color_scale = 1.0 / 255.0;
    auto pixel = data + j * bytes_per_scaneline + i * bytes_per_pixel;

    return Color(color_scale * pixel[0], color_scale * pixel[1], color_scale * pixel[2]);
}
*/