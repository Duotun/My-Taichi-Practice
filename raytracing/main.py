import taichi as ti
import numpy as np
from camera import Camera
from ray import Ray
import vector   #lots of utility functions
from hittable import *  #import all related stuff

ti.init(arch=ti.gpu)

# global variables
aspect_ratio = 16.0/9.0;
image_width = 400;
image_height = int(image_width/aspect_ratio);
pixels = ti.Vector.field(3, dtype=ti.f32, shape = (image_width, image_height));
cam = Camera();

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
        u = float(i)/(image_width - 1);
        v = float(j)/(image_height - 1);
        ray = cam.get_ray(u, v);
        pixels[i,j] = ray_color_background(ray);

#background_images
@ti.func 
def ray_color_background(ray):
    unit_dir = vector.unit_vector(ray.direction);
    t = 0.5 * (unit_dir[1]+1.0);
    return (1.0 - t) * vector.WHITE + t * vector.BLUE;       

#define the main function
def main():
    render_image();
    ti.tools.imwrite(pixels.to_numpy(), 'out.png');
    


if __name__ == '__main__':
    main();
    