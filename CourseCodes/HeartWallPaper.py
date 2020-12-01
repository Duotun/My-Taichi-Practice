import taichi as ti
import numpy as np

ti.init(debug=False, arch = ti.gpu)

Width, Hight = 960, 540
NUM_HEARTS =50.
LIGHT_DIR = ti.Vector([0.577,0.577,-0.577])
Heart_Color = ti.Vector([1.,0.05,0.05])
pixels = ti.Vector.field(4,dtype=ti.f32, shape=(Width, Hight))   # used as fragColor here from the shadertoy


#parameters to control the heart
r = 0.3;

@ti.func
def clamp(x, a_min, a_max):
    return min(max(x,a_min),a_max)

@ti.func
def smoothstep(a, b, t):
    p = clamp((t-a)/(b-a),0.,1.)
    v = p*p*(3.-2.*p)
    return v

@ti.func
def mix(x, y, a):
    return x* (1.0 -a) + y*a

@ti.func
def smoothmax(a, b, k): # k is the step
    h = clamp(0.5+0.5*(b-a/k),0.,1.)
    return mix(a,b,h) + k*h*(1.0-h)*0.5;  

@ti.func
def Heart(uv,b):
    r = 0.25
    b*= r
    uv[0]*=0.7;
        #make it smooth
    uv[1]-= smoothmax(ti.sqrt(ti.abs(uv[0]))*0.5,b,0.12)   # ad uv to make heart0like shape
        #compensate for the shift
    uv[1]+=0.1+0.5*b
    d = uv.norm()
    return smoothstep(r+b,r-b-0.01,d)  #tight circle


@ti.kernel
def Renderer(t: ti.f32,b:ti.f32):
    col = ti.Vector([0.,0.,0.,1.])
    for i, j in pixels:   # simulate the shadertoy method in MainImage()
        uv = ti.Vector([i - 0.5 * Width, j - 0.5*Hight])/min(Width, Hight)
        c = Heart(uv,b)* Heart_Color;
        col = ti.Vector([c[0],c[1],c[2],1.])
        pixels[i,j] = col;


gui = ti.GUI("Heart WallPaper", res = (Width, Hight))
b = 0.01
for itime in range(1000000):
    mousecoord = ti.Vector(gui.get_cursor_pos())
    for e in gui.get_events():
        if e.key == ti.GUI.ESCAPE:
            exit()
        if e.key == ti.GUI.MOVE or ti.GUI.WHEEL :
            #print(mousecoord[1])
            b = mousecoord[1]*0.3;
    Renderer(itime *0.03,b)   # simulate 30 fps ->render 
    gui.set_image(pixels.to_numpy())
    #process the mouse coordiante
    gui.show()
