/*
 * Utility Functions for Color class
 * Note - Color is just an alias for
 * a Vec3 class
 */
#ifndef COLOR_H
#define COLOR_H

#include <iostream>
#include "Vec3.h"

// Write RGB value of color vector
__host__ void writeColor(std::ostream &out, Color pixelColor, int samples_per_pixel);

#endif // COLOR_H