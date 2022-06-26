import taichi as ti
import taichi_glsl as ts
import math   #utilize pi


tiColor = ts.vec3;
WHITE = tiColor(1.0, 1.0, 1.0);
BLUE = tiColor(0.5, 0.7, 1.0);
RED = tiColor (1.0, 0.0, 0.0)

#for parallel computing
@ti.func
def unit_vector(v):
    v_n = v;
    return  v_n.normalized();

#sampline functions
