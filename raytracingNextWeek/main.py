import taichi as ti
import numpy as np
from camera import Camera
from ray import Ray
import vector   #lots of utility functions
from hittable import *  #import all related stuff
import material
import time
import random

#for the cornell box scene, 600, 1.0
# for the cover scene, 400, 16.0/9.0
# I just add rectangles and lights and then go to the restoflife section
ti.init(arch=ti.gpu, kernel_profiler=True, device_memory_fraction=0.4) 

cam_aspect_ratio = 16.0/9.0;
image_width = 1200;
image_height = int(image_width/cam_aspect_ratio);
pixels = ti.Vector.field(3, dtype=ti.f32, shape = (image_width, image_height));   # may changed with the later scene settings

#scene parameters
look_from = vec3f(13, 2, 3);
look_at = vec3f(0, 0, 0);
focus_dist_in = 10.0
cam = Camera(look_from, look_at, vec3f(0, 1, 0), 20.0, aspect_ratio = cam_aspect_ratio, aperture=0.1, focus_dist= focus_dist_in);
world = hittable_list();
samples_per_pixel = 100;
max_depth = 50;

backgroundColor = vec3f(0, 0, 0);

#rendering
@ti.kernel
def render_image():
    ti.loop_config(parallelize=8, block_dim=128)
    for i, j in pixels:
        pix_color = ti.Vector([0.0, 0.0, 0.0]);
        for n in range(samples_per_pixel):
            u = float(i + vector.random_number())/(image_width - 1);
            v = float(j + vector.random_number())/(image_height - 1);
            ray = cam.get_ray(u, v);
            pix_color += ray_color(ray, max_depth);
            #pix_color += ray_color_normal(ray);
        # perform the gamma correction in the end (1/2)
        pixels[i,j] = ti.sqrt(pix_color / samples_per_pixel);
        

#background_images
@ti.func 
def ray_color_background(ray):
    unit_dir = vector.unit_vector(ray.direction);
    t = 0.5 * (unit_dir[1]+1.0);
    return (1.0 - t) * vector.WHITE + t * vector.BLUE;  

@ti.func
def ray_background(color):
    return color;

@ti.func
def ray_color(ray, depth, background):   #background is the passed in color
    tmp_color = ti.Vector([0.0, 0.0, 0.0]);  #return black if the max depth reaches
    attenuation_color = ti.Vector([1.0, 1.0, 1.0]);
    for n in range(depth):
        is_hit, rec, mat = world.hit(ray, 0.001, 10e8);  
        if is_hit == True:
            is_scatter, out_dir, changed_color = mat.scatter(ray.direction, rec);
            emitted_color = mat.emit(rec);
            if is_scatter == True:
                tm_in = ray.time();
                ray =  Ray(origin = rec.pos, direction= out_dir, tm=tm_in);
                attenuation_color = emitted_color + attenuation_color * changed_color;
            else:
                tmp_color = emitted_color * attenuation_color;
                break;
        else:
            tmp_color = ray_background(background) * attenuation_color;    #+= consider the emitted color
    return tmp_color;


def cornel_box_Scene():
    global image_width
    global image_height
    global samples_per_pixel
    global max_depth
    global cam 
    global world 
    global backgroundColor
    global pixels 

    #Scene
    samples_per_pixel = 400;
    max_depth = 50;
    image_width = 600;  
    cam_aspect_ratio = 1.0
    image_height = int(image_width/cam_aspect_ratio);
    pixels = ti.Vector.field(3, dtype=ti.f32, shape = (image_width, image_height));

    mat_red = material._Material(color = vec3f(0.65, 0.05, 0.05), matindex = 0, roughness=0.0, ior = 0.0);
    mat_white = material._Material(color = vec3f(0.73, 0.73, 0.73), matindex = 0, roughness=0.0, ior = 0.0);
    mat_green = material._Material(color = vec3f(0.12, 0.45, 0.15), matindex = 0, roughness=0.0, ior = 0.0);
    mat_light = material._Material(color = vec3f(15, 15, 15), matindex = 3, roughness=0.0, ior = 0.0);

    world.add(yz_rect(0,555, 0, 555, 555, mat_green));
    world.add(yz_rect(0,555, 0, 555, 0, mat_red));
    world.add(xz_rect(213, 343, 227, 332, 554, mat_light));
    world.add(xz_rect(0,555, 0, 555, 0, mat_white));
    world.add(xz_rect(0,555, 0, 555, 555, mat_white));
    world.add(xy_rect(0,555, 0, 555, 555, mat_white));

    #camera
    look_from = vec3f(278, 278, -800.0);
    look_at = vec3f(278, 278, 0.0);
    vup = vec3f(0.0, 1.0, 0.0);
    dist_to_focus = 10.0;
    aperture = 0.0;

    cam = Camera(look_from, look_at, vup, 40, cam_aspect_ratio, aperture, dist_to_focus);


def random_scene():
    #Images
    global image_width
    global image_height
    global samples_per_pixel
    global max_depth
    global cam 
    global world
    global backgroundColor
    global pixels 

    cam_aspect_ratio = 3.0/2.0; 
    image_width = 1200;  
    image_height = int(image_width/cam_aspect_ratio);
    samples_per_pixel = 100;
    max_depth = 50;
    image_height = int(image_width/cam_aspect_ratio);
    pixels = ti.Vector.field(3, dtype=ti.f32, shape = (image_width, image_height));

    #construct the world
    ground_material = material._Material(color = vec3f(0.5, 0.5, 0.5), matindex = 0, roughness=0.0, ior = 0.0);
    world.add(Sphere(ti.Vector([0, -1000.0, 0]), 1000, ground_material));

    static_point = vec3f(4.0, 0.2, 0.0);

    mat1 = material._Material(color = vec3f(0.8, 0.8, 0.8), matindex =2, roughness=0.0, ior=1.5);  # dielectric
    mat2 = material._Material(color = vec3f(0.5, 0.5, 0.5), matindex = 0, roughness=0.0, ior = 0.0);  # diffuse
    mat3 = material._Material(color = vec3f(0.7, 0.6, 0.5), matindex = 1, roughness =0.0, ior =0.0);  # metal
    for a in range(-3, 3):
        for b in range(-3, 3):
            choose_mat = random.random();
            center = vec3f(a+0.9*random.random(), 0.2, b+0.9*random.random());
            mat = material._Material(color = vec3f(0.7, 0.6, 0.5), matindex = 1, roughness =0.0, ior =0.0);
            if((center - static_point).norm() > 0.9):    
                if choose_mat <0.8:
                    tmpcolor = vec3f(random.random()*random.random(), random.random()*random.random(), random.random()*random.random());
                    mat = material._Material(color = tmpcolor, matindex = 0, roughness=0.0, ior = 0.0);
                    center2 = center + vec3f(0, random.random()*0.5, 0.0);
                    world.add(Moving_Sphere(center, center2, 0.2, mat, 0.0, 1.0));
                elif choose_mat < 0.95:
                    tmpcolor = vec3f(random.random()*0.5 +0.5,  random.random()*0.5 +0.5, random.random()*0.5 +0.5);
                    mat = material._Material(color = tmpcolor, matindex = 1, roughness=0.0, ior = 0.0);
                    world.add(Sphere(center, 0.2, mat));
                else:
                    mat = material._Material(color = vec3f(1.0, 1.0, 1.0), matindex = 2, roughness=0.0, ior = 1.5);  # dielectric
                    world.add(Sphere(center, 0.2, mat));
            

    world.add(Sphere(vec3f(0.0, 1.0, 0.0), 1.0, mat1));
    world.add(Sphere(vec3f(-4.0, 1.0, 0.0), 1.0, mat2));
    world.add(Sphere(vec3f(4.0, 1.0, 0.0), 1.0, mat3));
    #camera
    look_from = vec3f(13, 2, 3);
    look_at = vec3f(0.0, 0.0, 0.0);
    vup = vec3f(0.0, 1.0, 0.0);
    dist_to_focus = 1.0;
    aperture = 0.1;

    cam = Camera(look_from, look_at, vup, 20, cam_aspect_ratio, aperture, dist_to_focus, 0.0, 1.0);


def DiffuseLight_Scene():
    global image_width
    global image_height
    global cam_aspect_ratio
    global samples_per_pixel
    global max_depth
    global cam 
    global world 
    global backgroundColor
    global pixels

    image_width = 1200;  # or 400
    cam_aspect_ratio = 16.0/9.0;
    image_height = int(image_width/cam_aspect_ratio);
    pixels = ti.Vector.field(3, dtype=ti.f32, shape = (image_width, image_height));
    samples_per_pixel = 400;
    max_depth = 30;
    backgroundColor = vec3f(0, 0, 0);

    mat_diffuse = material._Material(color = vec3f(0.0, 0.5, 0.5), matindex = 0, roughness=0.0, ior = 0.0);
    mat_light = material._Material(color = vec3f(4, 4, 4), matindex = 3, roughness=0.0, ior = 0.0);

    world.add(Sphere(vec3f(0.0, -1000, 0.0), 1000.0, mat_diffuse));
    world.add(Sphere(vec3f(0.0, 2, 0.0), 2, mat_diffuse));
    world.add(xy_rect(3, 5, 1, 3, -2, mat_light));

    #camera
    look_from = vec3f(26, 3, 6);
    look_at = vec3f(0.0, 2.0, 0.0);
    vup = vec3f(0.0, 1.0, 0.0);
    dist_to_focus = 1.0;
    aperture = 0;

    cam = Camera(look_from, look_at, vup, 20, cam_aspect_ratio, aperture, dist_to_focus, 0.0, 1.0);


@ti.kernel
def Render_Pass():
    for i, j in pixels:
        u = float(i + vector.random_number())/(image_width - 1);
        v = float(j + vector.random_number())/(image_height - 1);
        ray = cam.get_ray(u, v);
        # perform the gamma correction in the end (1/2)
        pixels[i,j] += ray_color(ray, max_depth, background=backgroundColor); 

def get_buffer(samples):
    return (pixels.to_numpy()/ samples)**0.5;

def main():
    # -- Render with Seriliazed Samples
    #random_scene();
    #cornell_BoxScene();
    #DiffuseLight_Scene();
    cornel_box_Scene();
    #gui = ti.GUI("Ray Tracing in One Weekend", res=(image_width, image_height));
    start_time = time.time();
    for i in range(samples_per_pixel):
        Render_Pass();
        
    #    gui.set_image(get_buffer(samples_per_pixel));
    #    gui.show();
    
    #perform the gamma correction in the end
    pixel_colors = (pixels.to_numpy()/ samples_per_pixel)**0.5;
    ti.tools.imwrite(pixel_colors, 'out.png');
    #gui.set_image(pixel_colors);
    #gui.show("out_t.png");

    ti.profiler.print_kernel_profiler_info();
    print("--- Time Elapsed: --- %s seconds" %(time.time() - start_time));


if __name__ == '__main__':
    main();