/*
 * Utility Functions for Color class
 * Note - Color is just an alias for
 * a Vec3 class
 */

#include <iostream>
#include "Color.h"
#include "shader_consts.h"

void writeColor(std::ostream &out, Color pixel_color, int samples_per_pixel)
{
    // Get RGB values
    float r = pixel_color.x();
    float g = pixel_color.y();
    float b = pixel_color.z();

    // Divide color by number of samples and gamma-correct for gamma = 2.0.
    float scale = 1.0 / samples_per_pixel;
    r = sqrt(r * scale);
    g = sqrt(g * scale);
    b = sqrt(b * scale);

    // Print out RGB in [0, 255] form
    out << static_cast<int>(256 * clamp(r, 0.0, 0.99)) << " "
        << static_cast<int>(256 * clamp(g, 0.0, 0.99)) << " "
        << static_cast<int>(256 * clamp(b, 0.0, 0.99)) << "\n";
}