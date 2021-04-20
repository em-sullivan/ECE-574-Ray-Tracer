/*
 * This file contains a collection
 * of worlds/scenese in the ray tracer. Each world has
 * a world function and a camera function
 */

#ifndef WORLDS_H
#define WORLDS_H

#include "shader_consts.h"
#include "Hittable_List.h"
#include "Sphere.h"
#include "Camera.h"
#include "Material.h"
#include "Moving_Sphere.h"
#include "Perlin.h"
#include "Aarect.h"
#include "Box.h"
#include "Translate.h"
#include "Constant_Medium.h"
#include "Bvh_Node.h"

Hittable_List random_balls();
Camera random_balls_cam(float aspect_ratio);

Hittable_List three_balls();
Camera three_balls_cam(float aspect_ratio);

Hittable_List two_bit_balls();
Camera two_bit_balls_cam(float aspect_ratio);

Hittable_List two_fuzzy_balls();
Camera two_fuzzy_balls_cam(float aspect_ratio);

Hittable_List earf();
Camera earf_cam(float aspect_ratio);

Hittable_List simple_light();
Camera simple_light_cam(float aspect_ratio);

Hittable_List cornell_box();
Camera cornell_box_cam(float aspect_ratio);

Hittable_List cornell_smoke();
Camera cornell_smoke_cam(float aspect_ratio);

Hittable_List final_scene();
Camera final_scene_cam(float aspect_ratio);

Hittable_List solar_system();
Camera solar_system_cam(float aspect_ratio);

Hittable_List glow_balls();
Camera glow_balls_cam(float aspect_ratio);


#endif // WORLDS_H