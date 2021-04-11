/*
 * Functions for rendering an image
 * with the ray tracer
 */

#include "Vec3.h"
#include "Camera.h"
#include "Hittable_List.h"

#ifndef RENDER_H
#define RENDER_H

// Write RGB value of color vector
void saveColor(std::ostream &out, Color pixelColor, int samples_per_pixel);

// Render image, store in color array
void render(Color *image, int height, int width, int spp, int depth,
    Camera &cam, Hittable_List &world, Color &background);

// Save pixels to a PPM file
void saveImage(std::ostream &out, Color *pixels, int height, int width, int spp);

#endif // RENDER_H