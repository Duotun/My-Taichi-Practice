import taichi as ti
import numpy as np
from camera import Camera
from ray import Ray
import vector   #lots of utility functions
from hittable import *  #import all related stuff
import material

ti.init(arch=ti.gpu)

# global variables, image parameters
cam_aspect_ratio = 16.0/9.0;
image_width = 400;
image_height = int(image_width/cam_aspect_ratio);
pixels = ti.Vector.field(3, dtype=ti.f32, shape = (image_width, image_height));

#scene parameters
look_from =  vec3f(3, 3, 2);
look_at = vec3f(0, 0, -1);
focus_dist_in = (look_from - look_at).norm();
cam = Camera(look_from, look_at, vec3f(0, 1, 0), 20.0, aspect_ratio = cam_aspect_ratio, aperture=2.0, focus_dist= focus_dist_in);
world = hittable_list();
samples_per_pixel = 100;
max_depth = 50;


#test kernels, remember no field could be assigned in the kernels, they need to be allocated in the cpu side
@ti.kernel
def test_visual():
    for i, j in pixels:
        for k in ti.static(range(3)):  # access index of vector must be at compile time
            pixels[i, j][k] = (i+j) / 512.0;
            
@ti.kernel
def basic_image():
    for i, j in pixels:
        pixels[i, j][0] = float(i)/(image_width-1);
        pixels[i, j][1] = float(j)/(image_height-1);
        pixels[i, j][2] = 0.25;

#rendering
@ti.kernel
def render_image():
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
def ray_color_normal(ray):
    is_hit, rec = world.hit(ray, 0.001, 10e8);   # a fake max value
    tmp_color = ti.Vector([0.0, 0.0, 0.0]);
    if is_hit:
        tmp_color = 0.5*(rec.normal + vector.WHITE);
    else:
        tmp_color = ray_color_background(ray);     
    return tmp_color;

@ti.func
def ray_color(ray, depth):
    tmp_color = ti.Vector([0.0, 0.0, 0.0]);  #return black if the max depth reaches
    attenuation_color = ti.Vector([1.0, 1.0, 1.0]);
    for n in range(depth):
        is_hit, rec, mat = world.hit(ray, 0.0001, 10e8);  
        if is_hit == True:
            is_scatter, out_dir, changed_color = mat.scatter(ray.direction, rec);
            if is_scatter == True:
                ray =  Ray(origin = rec.pos, direction= out_dir);
                attenuation_color *= changed_color;
            else:
                tmp_color *=0.0;
                break;
        else:
            tmp_color = ray_color_background(ray) * attenuation_color;     
    return tmp_color;

#define the main function
#matindex = 0 - diffuse, matindex = 1 - metal
def main():
    #prepare the 3D world
    material_ground = material._Material(color = vec3f(0.8, 0.8, 0.0), matindex = 0, roughness=0.0, ior =0.0);
    material_center = material._Material(color =vec3f(0.1, 0.2, 0.5), matindex =0, roughness=0.0, ior=0.0);  #diffuse
    
    material_left = material._Material(color = vec3f(0.8, 0.8, 0.8), matindex =2, roughness=0.0, ior=1.5);  # dielectric
    material_right = material._Material(color = vec3f(0.8, 0.6, 0.2), matindex =1, roughness=0.0, ior=0.0);  #metal
    
    world.add(Sphere(ti.Vector([0.0, 0.0, -1.0]), 0.5, material_center));
    #world.add(Sphere(ti.Vector([0.0, 0.0, -1.0]), 0.5, material_left));
    world.add(Sphere(ti.Vector([0.0, -100.5, -1.0]), 100, material_ground));
    world.add(Sphere(ti.Vector([-1.0, 0.0, -1.0]), 0.5, material_left));
    world.add(Sphere(ti.Vector([-1.0, 0.0, -1.0]), -0.4, material_left));
    world.add(Sphere(ti.Vector([1.0, 0.0, -1.0]), 0.5, material_right));
    
    #print(type(world.objects[0]))
    render_image();
    ti.tools.imwrite(pixels.to_numpy(), 'out.png');
    


if __name__ == '__main__':
    main();
    