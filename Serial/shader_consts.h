/*
 * Constants and declaraitons
 * for this simple shader program
 */

#ifndef SHADER_CONSTS_H
#define SHADER_CONSTS_H

#include <cmath>
#include <limits>
#include <memory>
#include <cstdlib>

// Usings
using std::shared_ptr;
using std::make_shared;
using std::sqrt;

// Constants
#define INF std::numeric_limits<double>::infinity()
#define PI 3.1415926535897932385

// Utility function to convert degrees to rads
inline double deg_to_rad(double degrees)
{
    return degrees * PI / 180.0;
}

inline double random_double()
{
    // Return a (pseduo) random double between 0.0 and 1.0
    return rand() / (RAND_MAX + 1.0);
}

inline double random_double(double min, double max)
{
    // Return a (pseduo) random double between the min and max value
    return min + (max - min) * random_double();
}

inline double clamp(double x, double min, double max)
{
    if (x < min) return min;
    if (x > max) return max;
    return x;
}

// Some common headers
//#include "Ray.h"
//#include "Vec3.h"

#endif // SHADER_CONSTS_H