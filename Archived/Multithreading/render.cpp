/*
 * Functions for rendering garbage
 */

#include "render.h"
#include "omp.h"

Color rayColor(const Ray &r, const Color& background, const Hittable &world, int depth)
{
    hit_record rec;

    // If the ray bounce limit is exceeded, no more light is gathered
    if (depth <= 0)
        return Color(0, 0, 0);

    // If the ray didn't hit anything, return the background
    if (!world.hit(r, 0.001f, INF, rec)) {
        return background;
    }
    
    Ray scattered;
    Color attenuation;
    Color emitted = rec.mat_ptr->emitted(rec.u,rec.v,rec.p);

    if (rec.mat_ptr->scatter(r, rec, attenuation, scattered)) {
        return emitted + attenuation * rayColor(scattered, background, world, depth-1);
    }
    
    return emitted;
}

void saveColor(std::ostream &out, Color pixel_color, int samples_per_pixel)
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

void render(Color *image, int height, int width, int spp, int depth,
    Camera &cam, Hittable_List &world, Color &background)
{

    int i, j, s;
    Color pixel_color;
    Ray r;
    float u, v;

#pragma omp parallel shared(image, height, width, spp, depth, cam, world, background) private(j, i, s, pixel_color, r, u, v)
{

    #pragma omp for schedule(static) nowait
    for (j = height - 1; j >= 0; j--) {
        //std::cerr << "\rScaneline remaing: " << j << ' ' << std::flush;

        for (i = 0; i < width; i++) {
            pixel_color = Color(0, 0, 0);
            for (s = 0; s < spp; s++) {
                u = (i + random_float()) / (width - 1);
                v = (j + random_float()) / (height - 1);
                r = cam.get_ray(u, v);
                pixel_color += rayColor(r, background, world, depth);
            }

            image[j * width + i] = pixel_color;
        }
    }
}
}

void saveImage(std::ostream &out, Color *pixels, int height, int width, int spp)
{
    out << "P3\n" << width << " " << height << "\n255\n";

    for (int j = height - 1; j>= 0; j--) {
        for (int i = 0; i < width; i++) {
            saveColor(out, pixels[j * width + i], spp);
        }
    }
}