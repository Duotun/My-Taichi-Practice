import taichi as ti
import vector
#use class -> data-oriented


#utilize ti.dataclass to construc the taichi type in the taichi function
@ti.dataclass
class Ray:
    origin: vector.vec3f;
    direction: vector.vec3f;
    tm: ti.f32;
    
    @ti.func  
    def at(self, t):
        return self.origin + t* self.direction;


    @ti.func
    def time(self):
        return self.tm;
    
