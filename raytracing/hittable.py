from numpy import dtype
import taichi as ti
from ray import Ray

vec3f = ti.types.vector(3, ti.f32);
hit_record = ti.types.struct(
    pos = vec3f, normal=vec3f, t = ti.f32, 
)


@ti.data_oriented
class hittable:
    def __init__(self):
        pass
    #use pass to represent an abstruct class
    @ti.func
    def hit(self, ray, t_min, t_max):
        pass

@ti.data_oriented
class Sphere (hittable):
    def __init__(self, s_center, s_radius):
        self.center = s_center;
        self.radius = s_radius;
        
    #return the hit information
    @ti.func
    def hit(self, ray, t_min, t_max):
        is_hit = False;
        is_frontface = False;
        rec = hit_record.field(shape=());
        oc = ray.origin - self.center;
        a = ray.direction.dot(ray.direction);
        b = 2.0 * oc.dot(ray.direciton);
        c = oc.dot(oc) - self.radius * self.radius;
        discreminant = b*b - 4*a*c;
        
        return is_hit, rec 
        