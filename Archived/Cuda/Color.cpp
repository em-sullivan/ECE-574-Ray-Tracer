/*
 * Utility Functions for Color class
 * Note - Color is just an alias for
 * a Vec3 class
 */

#include <iostream>
#include "Color.h"
#include "shader_consts.h"

__host__ void writeColor(std::ostream &out, Color pixel_color)// int samples_per_pixel)
{
    // Get RGB values
   // float r = pixel_color.x();
    //float g = pixel_color.y();
    //float b = pixel_color.z();

    // Divide color by number of samples and gamma-correct for gamma = 2.0.
    //loat scale = 1.0 / samples_per_pixel;
    //r = sqrtf(r * scale);
    //g = sqrtf(g * scale);
    //b = sqrtf(b * scale);

    // Print out RGB in [0, 255] form
    out << static_cast<int>(256 * clamp(pixel_color.x(), 0.0, 0.99)) << " "
        << static_cast<int>(256 * clamp(pixel_color.y(), 0.0, 0.99)) << " "
        << static_cast<int>(256 * clamp(pixel_color.z(), 0.0, 0.99)) << "\n";
}