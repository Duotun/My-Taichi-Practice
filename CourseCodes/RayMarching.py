import taichi as ti
import numpy as np

ti.init(debug=False, arch = ti.gpu)

Width, Hight = 960, 540
MAX_STEPS =100
MAX_DIST =100.
SURF_DIST = 0.001

pixels = ti.Vector.field(4,dtype=ti.f32, shape=(Width, Hight))   # used as fragColor here from the shadertoy

@ti.func
def Rot(angle):
    s = ti.sin(angle)
    c = ti.cos(angle)
    return ti.Matrix([[c,-s],[s,c]])

@ti.func
def mix(x, y, a):
    return x* (1.0 -a) + y*a

@ti.func
def smoothmin(a, b, k): # k is the step
    h = clamp(0.5+0.5*(b-a/k),0.,1.)
    return mix(b,a,h) - k*h*(1.0-h)  

@ti.func
def sdfsphere(p, center, radius):
    d = p - center;
    dis = d.norm() - radius;
    return dis;

@ti.func
def sdfBox(p,length):
    p = ti.abs(p) - length;
    d = ti.max(p,ti.Vector([0.,0.,0.])).norm();
    d += ti.min(ti.max(p[0],ti.max(p[1],p[2])),0.);
    return d;
    
@ti.func
def sdfCapsule(p,a,b,radius):
    ab = b-a;
    ap = p-a;
    t = ab.dot(ap) / ab.dot(ab)
    t = clamp(t,0.,1.)
    c = a+t*ab;
    d = (p-c).norm() - radius;
    return d;

@ti.func
def sdfTorus(p,center,r):
    p = p-center;
    x = ti.Vector([p[0],p[2]]).norm() - r[0]
    return ti.Vector([x,p[1]]).norm() - r[1];

@ti.func
def sdfscene(p,t=0):
    center = ti.Vector([-3.,1,6.])
    radius = .5;
    planesdf = p.y
    spheresdf = sdfsphere(p,center,radius)
    torussdf = sdfTorus(p,ti.Vector([0,0.5,7]),ti.Vector([1.58,.2]))
    capsulesdf = sdfCapsule(p,ti.Vector([0.4,0.3,6]),ti.Vector([0.7,1.5,6.]),.2); 

    bp = p - ti.Vector([-3.4,0.5,6])  # translation
    #bp -= ti.Vector([0.,0.75,3])
    tmpbp = ti.Vector([bp[0],bp[2]]);
    tmpbp = Rot(t)@tmpbp;
    bp[0] = tmpbp[0]
    bp[2] = tmpbp[1]
    boxsdf = sdfBox(bp,ti.Vector([0.5,0.5,0.5]));

    d = smoothmin(boxsdf,spheresdf,0.4);
    d = ti.min(planesdf,d);
    #d = ti.min(sdfBox(p,ti.Vector([-3.,0.5,6.]),ti.Vector([0.5,0.5,0.5])),d);
    #d = ti.min(sdfBox(p,ti.Vector([-3.,1.5,5.4]),ti.Vector([0.5,0.5,0.5])),d);
    d = ti.min(torussdf,d);
    d = ti.min(capsulesdf,d);


    #blend
    bp = p - ti.Vector([3.5,0.5,6])
    dbox = sdfBox(bp,ti.Vector([0.5,0.5,0.5]))
    dball = sdfsphere(p,ti.Vector([3.5,0.5,6]),radius);
    blend = mix(dbox,dball,ti.sin(t)*.5+.5)
    d = ti.min(blend,d)
    return d;

@ti.func
def clamp(x, a_min, a_max):
    return min(max(x,a_min),a_max)

@ti.func
def GetNormal(p,t):
    d = sdfscene(p,t)
    ex = ti.Vector([0.001, 0., 0.])
    ey = ti.Vector([0., 0.001, 0.])
    ez = ti.Vector([0., 0., 0.001])
    n = ti.Vector([d,d,d]) - ti.Vector([sdfscene(p-ex,t),sdfscene(p-ey,t),sdfscene(p-ez,t)])
    return n.normalized();


@ti.func
def GetLight(t, p):  #t is the time
    lightPos = ti.Vector([1.,5.,6.])
    #lightPos[0] += ti.sin(t)
    #lightPos[2] += ti.cos(t)
    l = (lightPos -p).normalized();
    n = GetNormal(p,t)

    dif = clamp(n.dot(l)*0.5+0.5,0.,1.) 
    d = RayMarch(p+n*SURF_DIST*2.,l,t)
    if d<(lightPos - p).norm() and p[1]<0.001:
        dif *= 0.5
    return dif;


@ti.func  # must be type hinted
def RayMarch(ro,rd,t):
    d0= 0.0;
    p = ti.Vector([0.,0.,0.])
    for i in range(MAX_STEPS):
       p = ro + rd*d0;
       #print(p.data_type())
       ds = sdfscene(p,t);   
       d0+=ds;
       # to long or to small -> break
       if  d0>MAX_DIST or ds<SURF_DIST:
            break;
    return d0;


@ti.kernel
def Renderer(t: ti.f32):
    col = ti.Vector([0.,0.,0.,1.])
    for i, j in pixels:   # simulate the shadertoy method in MainImage()
        uv = ti.Vector([i - 0.5 * Width, j - 0.5*Hight])/min(Width, Hight)
        #col = ti.Vector(3,dt=ti.f32,shape =1)
        ro = ti.Vector([0.,3.,0.])
        rd = ti.Vector([uv[0],uv[1]-0.4,1.]).normalized()
        d = RayMarch(ro,rd,t)
        p = ro+rd*d
        dif = GetLight(t,p)
        col = ti.Vector([dif, dif, dif, 1.])
        #dif = 0.5  # currently pure color
        pixels[i,j]= col;


gui = ti.GUI("Ray Maching Primitive Shapes", res = (Width, Hight))
for itime in range(1000000):
    Renderer(itime *0.03)   # simulate 30 fps ->render 
    gui.set_image(pixels.to_numpy())
    gui.show()
