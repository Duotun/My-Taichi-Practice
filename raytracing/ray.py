import taichi as ti
import vector
#use class -> data-oriented

@ti.data_oriented
class Ray:
    def __init__(self):
        self.origin = ti.Vector.field(3, dtype = ti.f32);
        self.direction = ti.Vector.field(3, dtype = ti.f32);
    
    def __init__(self, org, dir):
        self.origin = org;
        self.direction = dir.normalized();
    
    @ti.func  
    def at(self, t):
        return self.origin + t* self.direction;

