import taichi as ti
import taichi_glsl as ts
import math   #utilize pi


tiColor = ts.vec3;
WHITE = tiColor(1.0, 1.0, 1.0);
BLUE = tiColor(0.5, 0.7, 1.0);
RED = tiColor (1.0, 0.0, 0.0)

vec3f = ti.types.vector(3, ti.f32)
#for parallel computing
@ti.func
def unit_vector(v):
    v_n = v;
    return  v_n.normalized();

#sample & random functions

@ti.func
def random_number():    
    return ts.rand();


@ti.func
def random_number_range(min, max):
    return ts.randRange(min, max);

@ti.func
def degrees_to_radians(degrees):
    return degrees * math.pi / 180.0;

@ti.func
def random_vector():
    return ti.Vector([random_number(), random_number(), random_number()]);

@ti.func
def random_vector_range(min, max):
    return ti.Vector([random_number_range(min, max), random_number_range(min, max), random_number_range(min, max)]);


@ti.func
def random_in_unit_sphere():
    p = random_vector_range(-1.0, 1.0);
    while True:
        if(p.norm_sqr() <1.0):
            break;
        p = random_vector_range(-1.0, 1.0);
    return p;

@ti.func
def random_unit_vector():
    return random_in_unit_sphere().normalized();

@ti.func
def random_hemisphere(normal):
    in_unit_sphere = random_in_unit_sphere();
    if(in_unit_sphere.dot(normal)<0.0):
        in_unit_sphere = -in_unit_sphere;
    return in_unit_sphere;

@ti.func
def random_in_unit_disk():
    p = vec3f(random_number_range(-1, 1), random_number_range(-1,1), 0.0);
    while True:
        if p.norm_sqr() < 1.0:
            break;
        p[0] = random_number_range(-1, 1);
        p[1] = random_number_range(-1, 1);
    return p;

#don't reflect is near to the zero surface
@ti.func
def near_zero(vec_e):
    return ti.abs(vec_e[0]) <1e-8 and  ti.abs(vec_e[1]) <1e-8  and ti.abs(vec_e[2]) <1e-8; 
