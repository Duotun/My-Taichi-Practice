import taichi as ti
from ray import Ray


@ti.data_oriented
class Camera:
    def __init__(self, aspect_ratio=16.0/9.0):
        self.viewport_height = 2.0
        self.viewport_width = aspect_ratio * self.viewport_height;
        self.focal_length = 1.0;
        self.vec3f = ti.types.vector(3, ti.f32);
        self.origin=self.vec3f(0, 0, 0);   #assign with array
        self.horizontal= self.vec3f(self.viewport_width, 0, 0);
        self.vertical = self.vec3f(0, self.viewport_height, 0);
        self.lower_left_corner = self.origin - self.horizontal/2.0 - self.vertical/2 - self.vec3f(0, 0, self.focal_length);
        self.reset();
        
    
    def reset(self):    
        self.origin = self.vec3f(0, 0, 0);   #assign with array
        self.horizontal= self.vec3f(self.viewport_width, 0, 0);
        self.vertical = self.vec3f(0, self.viewport_height, 0);
        self.lower_left_corner = self.origin - self.horizontal/2.0 - self.vertical/2 - self.vec3f(0, 0, self.focal_length);
        
        
    @ti.func
    def get_ray(self, u, v):
        r = Ray(self.origin, self.lower_left_corner + u*self.horizontal + v*self.vertical- self.origin);
        return r;
        