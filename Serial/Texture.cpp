/*
 * Texture class source file
 * Makes colors on surfaces procedural
 */

#include "Texture.h"

/*
 * Solid Color
 */
Solid_Color::Solid_Color(Color c)
{
    color_v = c;
}

Solid_Color::Solid_Color(float r, float g, float b)
{
    color_v = Color(r, g, b);
}

Color Solid_Color::value(float u, float v, const Point3 &p) const
{
    return color_v;
}

/*
 * Solid Checkard pattern
 */

Checkered::Checkered(shared_ptr<Texture> even, shared_ptr<Texture> odd)
{
    even_tiles = even;
    odd_tiles = odd;
}

Checkered::Checkered(Color c1, Color c2)
{
    even_tiles = make_shared<Solid_Color>(c1);
    odd_tiles = make_shared<Solid_Color>(c2);
}

Color Checkered::value(float u, float v, const Point3 &p) const
{
    auto sines = sin(10 * p.x()) * sin(10 * p.y()) * sin(10 * p.z());
    if (sines < 0)
        return odd_tiles->value(u, v, p);
    else
        return even_tiles->value(u, v, p);
}

Noise_Text::Noise_Text()
{
    scale = 1.0;
}

Noise_Text::Noise_Text(float sc)
{
    scale = sc;
}

Color Noise_Text::value(float u, float v, const Point3 &p) const
{
    // Generator noise
    // return Color(1, 1, 1) * noise.turb(p * scale);
    // Comments out straight noise, igves a marbel like effect
    return Color(1, 1, 1) * 0.5 * (1 + sin(scale * p.z() + 10 * noise.turb(p)));
}