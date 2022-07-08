import vector
import taichi as ti

#hmmm maybe no use for now
@ti.data_oriented
class texture:
    def __init__(self):
        pass;

    def color_value(self, u = 0.0, v=0.0, p=vector.vec3f(0.0, 0.0, 0.0)):
        pass;


@ti.data_oriented
class solid_color(texture):
    def __init__(self, color=vector.vec3f(1.0, 1.0, 1.0)):
        self.color = color;

    def color_value(self, u=0, v=0, p=vector.vec3f(0, 0, 0)):
        return self.color;
    
