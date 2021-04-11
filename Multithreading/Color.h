/*
 * Utility Functions for Color class
 * Note - Color is just an alias for
 * a Vec3 class
 * 
 * NOTE: This is just around for legacy reasons, this is handled in the render
 * file now lol
 */
#ifndef COLOR_H
#define COLOR_H

#include <iostream>
#include "Vec3.h"

// Write RGB value of color vector
void writeColor(std::ostream &out, Color pixelColor, int samples_per_pixel);

#endif // COLOR_H