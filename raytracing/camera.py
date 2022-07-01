import taichi as ti
from ray import Ray


@ti.data_oriented
class Camera:
    def __init__(self, aspect_ratio=16.0/9.0):
        self.viewport_height = 2.0
        self.viewport_width = aspect_ratio * self.viewport_height;
        self.focal_length = 1.0;
        self.origin= ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)   #assign with array
        self.horizontal= ti.Vector([self.viewport_width, 0, 0], dt=ti.f32);
        self.vertical = ti.Vector([0, self.viewport_height, 0], dt=ti.f32);
        self.lower_left_corner = self.origin - self.horizontal/2.0 - self.vertical/2 - ti.Vector([0, 0, self.focal_length]);
        self.reset();
        
    
    def reset(self):    
        self.origin= ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)   #assign with array
        self.horizontal= ti.Vector([self.viewport_width, 0, 0], dt=ti.f32);
        self.vertical = ti.Vector([0, self.viewport_height, 0],dt=ti.f32);
        self.lower_left_corner = self.origin - self.horizontal/2.0 - self.vertical/2 - ti.Vector([0, 0, self.focal_length]);
        
        
    @ti.func
    def get_ray(self, u, v):
        r = Ray(origin=self.origin, direction=self.lower_left_corner + u*self.horizontal + v*self.vertical- self.origin);
        return r;
        