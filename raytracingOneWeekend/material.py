from operator import ior
import taichi as ti
from vector import *


@ti.func
def reflect(v, n):
    return v - 2.0 * n.dot(v) * n;

@ti.func
def refract(v, n, etai_over_etat):
    cose_theta = min(-v.dot(n), 1.0);
    r_out_perp = etai_over_etat * (v + cose_theta * n);
    r_out_parallel = - ti.sqrt(abs(1.0 - r_out_perp.norm_sqr()))*n;
    return r_out_perp + r_out_parallel;

@ti.func
def reflectance(cosine, idx):
    # Schlick's approximation for reflectance
    r0 = ((1.0 - idx) / (1.0 + idx))**2;
    return r0 + (1.0-r0)*((1.0 - cosine)**5);
    

# I am defining a unified class struct for material 
# the scattering are different from material index, 0 -diffuse, 1 - metal, 2 - dielectric

@ti.struct_class
class _Material():
    color: vec3f;
    matindex: ti.i32;
    roughness :ti.f32;
    ior: ti.f32;
    
    @ti.func
    def scatter(self, dir_in, rec):
        is_scattered = False;
        out_dir = vec3f(0, 0, 0);
        attenuation_color = vec3f(0, 0, 0)
        if self.matindex == 0:  #diffuse
            is_scattered, out_dir, attenuation_color = self.scatter_diffuse(rec);
        
        if self.matindex == 1:  #metal
            is_scattered, out_dir, attenuation_color = self.scatter_metal(dir_in, rec);
            
        if self.matindex == 2: #dielectric
            is_scattered, out_dir, attenuation_color = self.scatter_dielectric(dir_in, rec);
            
        return is_scattered, out_dir, attenuation_color;
    
    @ti.func 
    def scatter_diffuse(self, rec):
        out_dir = random_hemisphere(rec.normal);
        if near_zero(out_dir):
            out_dir = rec.normal;
        attenuation_color = self.color;
        return True, out_dir, attenuation_color;
    
    @ti.func 
    def scatter_metal(self, dir_in, rec):
        reflected = reflect(dir_in, rec.normal + self.roughness * random_in_unit_sphere());
        is_scattered = False;
        is_scattered = reflected.dot(rec.normal) >0;
        attenuation = self.color;
        return is_scattered, reflected, attenuation; 
    
    @ti.func
    def scatter_dielectric(self, dir_in, rec):
        attenuation_color = vec3f(1.0, 1.0, 1.0);
        unit_dir = dir_in.normalized();
        refraction_ratio = self.ior;
        if rec.frontface == 1:    #if frontface
            refraction_ratio = (1.0/ self.ior);
        cos_theta = min(-unit_dir.dot(rec.normal), 1.0);
        sin_theta = ti.sqrt(1.0 - cos_theta * cos_theta);
        # check whether pure reflection
        out_dir = ti.Vector([0.0, 0.0, 0.0]);
        #out_dir = refract(unit_dir, rec.normal, refraction_ratio)
        cannot_refract = (refraction_ratio * sin_theta) >1.0;
        if cannot_refract or reflectance(cos_theta, refraction_ratio) > ti.random():
            out_dir = reflect(unit_dir, rec.normal);
        else:
            out_dir = refract(unit_dir, rec.normal, refraction_ratio);
        
        return True, out_dir, attenuation_color;
        

@ti.struct_class
class _Dielectric():
    color: vec3f;
    index: ti.f32;
    roughness: ti.f32;
    ior: ti.f32;




