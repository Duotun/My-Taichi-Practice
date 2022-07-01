import taichi as ti
from vector import *


@ti.struct_class
class _Diffuse():
    color: vec3f;
    index: ti.f32;
    roughness: ti.f32;
    ior: ti.f32;
    
    #define a static method no self needed
    @ti.func
    def scatter(self, dir_in, rec):
        out_dir = random_hemisphere(rec.normal);
        if near_zero(out_dir):
            out_dir = rec.normal;
        attenuation_color = self.color;
        return True, out_dir, attenuation_color;
    
    

@ti.struct_class
class _Metal():
    color: vec3f;
    index: ti.f32;
    roughness: ti.f32;
    ior: ti.f32;

@ti.struct_class
class _Dielectric():
    color: vec3f;
    index: ti.f32;
    roughness: ti.f32;
    ior: ti.f32;




