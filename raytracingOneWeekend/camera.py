from cmath import tan
import taichi as ti
from ray import Ray
from vector import random_in_unit_disk, unit_vector
from vector import vec3f

@ti.data_oriented
class Camera:
    def __init__(self, lookfrom, lookat, vup = ti.Vector([0.0, 1.0, 0.0], dt = ti.f32), vfov = 60.0, aspect_ratio=16.0/9.0, aperture=2.0, focus_dist = 6.0):
        self.vfov = vfov;
        theta = vfov * ti.math.pi / 180.0;
        h = ti.tan(theta/2.0);    #make sure it is a ti.float
        self.viewport_height = 2.0 * h
        self.viewport_width = aspect_ratio * self.viewport_height;
        
        self.lookfrom = lookfrom;
        self.lookat = lookat;
        self.focal_length = 1.0;
        
        self.origin= vec3f(0.0, 0.0, 0.0);   #assign with array
        self.horizontal= vec3f(0.0, 0.0, 0.0);  
        self.vertical = vec3f(0.0, 0.0, 0.0);  
        self.lower_left_corner = vec3f(0.0, 0.0, 0.0);  
        self.vup = vup;
        self.lens_radius = aperture/2.0;
        self.focus_dist = focus_dist;
        self.u = vec3f(0, 0, 0);
        self.v = vec3f(0, 0, 0);
        self.reset();
        
    
    def reset(self): 
        w = (self.lookfrom - self.lookat).normalized();
        self.u = (self.vup.cross(w)).normalized();
        self.v = w.cross(self.u);   
        self.origin= self.lookfrom;
        self.horizontal=  self.focus_dist * self.viewport_width * self.u;    # w, u ,v are the axes from the camera space
        self.vertical = self.focus_dist * self.viewport_height * self.v;
        self.lower_left_corner = self.origin - self.horizontal/2.0 - self.vertical/2.0 - self.focus_dist * w;
        
     
    @ti.func
    def get_ray(self, s, t):
        rd = self.lens_radius * random_in_unit_disk();
        offset = rd.x * self.u + rd.y * self.v;
        r = Ray(origin=self.origin + offset, direction=self.lower_left_corner + s*self.horizontal + t*self.vertical- self.origin - offset);
        #r = Ray(origin=self.origin, direction=self.lower_left_corner + u*self.horizontal + v*self.vertical- self.origin);
        return r;
        