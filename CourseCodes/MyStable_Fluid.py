# References
# # https://github.com/ShaneFX/GAMES201/tree/master/HW01
# https://github.com/taichi-dev/taichi/blob/3050606b44a64e3e1070835b7bfe22eee39a00a1/examples/stable_fluid.py


# use two advections -> update velocity (one material derivative) and dye property (another material derivative based on velocity)
# 2D Interactive Showcase
# remember the sensors in euler fields are stable
# bfecc-> advection, jacobi / conjugate gradient -> projection 

import taichi as ti 
import numpy as np 
import math
import time

res = 512
dt = 0.03
scale = 0.5  # +1, -1
dx = 2   # scale = 1/dx
p_jacobi_iters = 160
f_strength = 10000.0
dye_decay = 0.99   # for impulse
force_radius = res/3.0
conjugate_gradient = True

ti.init(debug=False, arch=ti.gpu)

_velocities = ti.Vector.field(2,float, shape = (res, res))
_intermedia_velocities = ti.Vector.field(2,float, shape=(res, res)) #for advection of bfecc methods
_new_velocities = ti.Vector.field(2, float, shape=(res,res))
velocity_divs = ti.field(float,shape=(res,res))
velocity_curls = ti.field(float, shape=(res, res))
_pressures = ti.field(float, shape=(res,res))
_new_pressures = ti.field(float, shape=(res,res))  # res is for the resolution
_dye_buffer = ti.Vector.field(3,float, shape=(res,res))  # 3 -> RGB
_intermedia_dye_buffer = ti.Vector.field(3, float, shape = (res,res))
_new_dye_buffer = ti.Vector.field(3, float, shape=(res,res))

class TexPair:
    def __init__(self, cur, nxt):
        self.cur = cur
        self.nxt = nxt
    
    def swap(self):   # for next step
        self.cur, self.next = self.nxt, self.cur

@ti.data_oriented
class Conjugate_gradient:
    def __init__(self, dim = 2, N =512, real= float):
        self.N = N
        self.dim = dim
        self.real = real

        self.N_tot = 2*self.N
        #I think don't need it for conjugate gradient
        self.N_ext = self.N //2   # floor dividion

        self.r = ti.field(dtype = self.real)   # r = b - Ax
        self.z = ti.field(dtype = self.real)   # currently no considering for the multi-grid
        self.x = ti.field(dtype=self.real)  # solution
        self.p = ti.field(dtype = self.real) #conjugate gradient  (symbol used from Wiki)
        self.Ap = ti.field(dtype = self.real)
        self.alpha = ti.field(dtype = self.real)
        self.beta = ti.field(dtype = self.real)  #using Non to access the index is because no shape here -> creat an axis
        self.sum = ti.field(dtype = self.real)

        indices = ti.ijk if self.dim ==3 else ti.ij
        self.grid = ti.root.pointer(indices, [self.N_tot // 4]).dense(indices, 4).place(self.x, self.p, self.Ap)
        self.grid = self.grid = ti.root.pointer(indices,[self.N_tot // (4 * 2**0)]).dense(indices,4).place(self.r, self.z) #only r, z, no restrictions


        ti.root.place(self.alpha,self.beta,self.sum)
        # #divide into two halves to initialize
    @ti.func
    def init_r(self, I, r_I):
        I = I + self.N_ext
        self.r[I] = r_I
        self.z[I] = 0.
        self.Ap[I] = 0.
        self.p[I] = 0.
        self.x[I] = 0.
    
    @ti.kernel 
    def init(self,r:ti.template(),k:ti.template()):
        #set up for the solver for  $\nabla^2 x = k r$, a scaled Poisson problem.
        for I in ti.grouped(ti.ndrange(*[self.N]*self.dim)):
            self.init_r(I,r[I]*k)
    
    @ti.func
    def get_x(self,I):
        I = I + self.N_ext
        return self.x[I]
    
    @ti.kernel
    def get_result(self, x:ti.template()):
        #obtain the solution field
        for I in ti.grouped(ti.ndrange(*[self.N] * self.dim)):
            x[I] = self.get_x(I)

    @ti.func
    def neighbor_sum(self,x,I):
        ret = ti.cast(0.0, self.real)
        for i in ti.static(range(self.dim)):
            offset = ti.Vector.unit(self.dim, i)
            ret += x[I + offset] + x[I - offset]
        return ret

    @ti.kernel
    def compute_Ap(self):
        for I in ti.grouped(self.Ap):
            self.Ap[I] = 2 * self.dim * self.p[I] - self.neighbor_sum(self.p, I)
    
    @ti.kernel  # for calculate the sum for residul, r^t *r
    def reduce(self, p: ti.template(), q: ti.template()):
        self.sum[None] = 0.
        for I in ti.grouped(p):
            self.sum[None] += p[I] *q[I]

    @ti.kernel
    def update_x(self):
        for I in ti.grouped(self.p):
            self.x[I] += self.alpha[None] * self.p[I]

    @ti.kernel
    def update_r(self):
        for I in ti.grouped(self.p):
            self.r[I] -= self.alpha[None] * self.Ap[I]

    @ti.kernel
    def update_p(self):   #update the conjugate
        for I in ti.grouped(self.p):
            self.p[I] = self.z[I] + self.beta[None] * self.p[I]

    def solve(self, max_iters = -1, eps = 1e-12, abs_tol = 1e-12, rel_tol = 1e-12, verbose = False):
        '''
Verbose is a general programming term for produce lots of logging output. 
You can think of it as asking the program to "tell me everything about what you are doing all the time". Just set it to true and see what happens.
'''
        # -1 (max_iterations) for no limite
        # eps used for avoiding zerodivisionError
        # abs_tol absolute tolerance of loss

        self.reduce(self.r,self.r)
        initial_rtr = self.sum[None]   #calculated from reduce method
        tol = max(abs_tol, initial_rtr*rel_tol)

        # self.r = b - Ax = b, since the initialization of x = 0
        # self.p = self.r 
        self.z.copy_from(self.r)
        self.update_p()
        old_ztr = self.sum[None];  # here is no difference

        #conjugate gradients begin
        iter = 0
        while max_iters ==-1 or iter < max_iters:
            #self.alpha = rtr / ptAp
            self.compute_Ap()
            self.reduce(self.p, self.Ap)
            pAp = self.sum[None]
            self.alpha[None] = old_ztr/(pAp +eps)


            #self.x = self.x + self.alpha * self.p
            self.update_x()

            #self.r = self.r - self.alpha * self.ap
            self.update_r()

            #check for convergence
            self.reduce(self.r,self.r)
            rtr = self.sum[None]

            if rtr<tol:
                break;
            
            self.z.copy_from(self.r)
            self.reduce(self.z,self.r)
            new_ztr = self.sum[None]
            self.beta[None] = new_ztr/(old_ztr+eps)
            
            #update p, self.p = self.z +self.beta * self.p
            self.update_p()
            old_ztr = new_ztr
            iter +=1

velocities_pair = TexPair(_velocities, _new_velocities)
pressures_pair = TexPair(_pressures, _new_pressures)
dyes_pair = TexPair(_dye_buffer, _new_dye_buffer)

if conjugate_gradient:
    cgg = Conjugate_gradient(dim=2, N=res)

@ti.func   #q is for the quantity q might be density, or velocity, or temperature, or many other things.)
def sample(qf, u, v): 
    I = ti.Vector([int(u),int(v)])
    I = max(0, min(res-1, I))  #element-wise comparison
    return qf[I]

@ti.func
def lerp(vl, vr, frac):
    #fraction [0.0, 1.0]
    return vl+ frac* (vr-vl)

@ti.func
def bilerp(vf, p):
    u, v = p
    s, t = u - 0.5, v - 0.5
    iu, iv = ti.floor(s), ti.floor(t)
    #frac
    fu, fv = s-iu, t-iv
    a = sample(vf, iu, iv)
    b = sample(vf, iu+1, iv)
    c = sample(vf, iu, iv+1)
    d = sample(vf, iu+1, iv+1)
    return lerp(lerp(a, b, fu),lerp(c, d, fu), fv)

@ti.func
def sample_minmax(vf, p):   # 2d case
    u, v = p
    s, t = u - 0.5, v - 0.5
    iu, iv = ti.floor(s), ti.floor(t)
    a = sample(vf, iu, iv)
    b = sample(vf, iu + 1, iv)
    c = sample(vf, iu, iv + 1)
    d = sample(vf, iu + 1, iv + 1)
    return min(a, b, c, d), max(a, b, c, d)

#the point of vacktrace is to utilize the velocity to obtain the material property of the last time step
@ti.func
def backtrace_rk1(vf: ti.template(), p, dt: ti.template()):
    p -= dt * bilerp(vf, p)
    return p

@ti.func
def backtrace_rk2(vf: ti.template(), p, dt: ti.template()):
    p_mid = p - 0.5*dt*bilerp(vf,p)
    p -= dt* bilerp(vf, p_mid)
    return p

@ti.func
def backtrace_rk3(vf: ti.template(), p, dt: ti.template()):
    v1 = bilerp(vf,p)
    p1 = p - 0.5*dt *v1
    v2 = bilerp(vf, p1)
    p2 = p - 0.75*dt*v2
    v3 = bilerp(vf,p2)
    p -= dt *((2.0/9.0) * v1 + (1.0/3.0)*v2 + (4.0/9.0)*v3)
    return p

backtrace = backtrace_rk3   # template to call

@ti.kernel   #open to external access
def advect_semilag(vf: ti.template(), qf:ti.template(), new_qf: ti.template(), intermedia_qf: ti.template()):
    for i, j in vf: #extract every sensor in the given resolution
        p = ti.Vector([i, j]) + 0.5
        p = backtrace(vf, p, dt)
        new_qf[i, j] = bilerp(qf, p)

@ti.kernel
def advect_bfecc(vf: ti.template(), qf: ti.template(), new_qf: ti.template(), intermedia_qf: ti.template()):
    for i, j in vf:  # here is the same as the semilag
        p = ti.Vector([i, j])+0.5
        p = backtrace(vf, p, dt)
        intermedia_qf[i, j] = bilerp(qf, p)
    
    for i, j in vf:
        p = ti.Vector([i, j]) +0.5
        p_two_star = backtrace(vf, p, -dt)  # for the reverse advection
        p_star = backtrace(vf, p, dt)   # for cutting 
        q_star = intermedia_qf[i,j]
        new_qf[i, j] = bilerp(intermedia_qf, p_two_star)
        new_qf[i, j] = q_star + 0.5 * (qf[i, j] - new_qf[i, j])

        #cutting
        min_val, max_val = sample_minmax(qf, p_star)
        cond = min_val < new_qf[i, j] < max_val
        for k in ti.static(range(cond.n)):  #use static to avoid some runtim overhead, .n means the number of components in the vector
            if not cond[k]:
                new_qf[i,j][k] = q_star[k]  #just use semi-lagrangian
        
advect = advect_bfecc

@ti.kernel
def apply_impulse(vf: ti.template(), dyef: ti.template(),
                  imp_data: ti.ext_arr()):
    for i, j in vf:
        omx, omy = imp_data[2], imp_data[3]
        mdir = ti.Vector([imp_data[0], imp_data[1]])
        dx, dy = (i + 0.5 - omx), (j + 0.5 - omy)
        d2 = dx * dx + dy * dy
        # dv = F * dt
        factor = ti.exp(-d2 / force_radius)
        momentum = mdir * f_strength * dt * factor
        v = vf[i, j]
        vf[i, j] = v + momentum
        # add dye
        dc = dyef[i, j]
        if mdir.norm() > 0.5:
            dc += ti.exp(-d2 * (4 / (res / 15)**2)) * ti.Vector(
                [imp_data[4], imp_data[5], imp_data[6]])
        dc *= dye_decay
        dyef[i, j] = dc


@ti.kernel
def divergence(vf: ti.template()):
    for i, j in vf:
        vl = sample(vf, i-1, j).x
        vr = sample(vf,i+1,j).x
        vb = sample(vf, i, j-1).y
        vt = sample(vf, i, j+1).y
        vc = sample(vf, i, j)
        if i==0:  #boundary
            vl = 0;
        if i ==res - 1:
            vr = 0
        if j ==0:
            vb =0
        if j == res - 1:
            vt = 0
        velocity_divs[i, j] = (vr - vl + vt - vb) * scale

@ti.kernel
def pressure_jacobi (pf: ti.template(), new_pf: ti.template()):
    for i, j in pf:
        pl = sample(pf, i-1, j)
        pr = sample(pf,i+1,j)
        pb = sample(pf, i, j-1)
        pt = sample(pf, i, j+1)
        div = velocity_divs[i, j]
        new_pf[i, j] = (pl+pr+pb+pt - div) * 0.25

@ti.kernel
def subtract_gradient(vf: ti.template(), pf: ti.template()):
    for i, j in vf:
        pl = sample(pf, i-1, j)
        pr = sample (pf, i+1, j)
        pb = sample(pf, i, j-1)
        pt = sample(pf, i, j+1)
        vf[i, j] -= scale * ti.Vector([pr - pl, pt - pb])

@ti.kernel
def vorticity(vf: ti.template()):
    for i, j in vf:   # cross product of velocity and divergence, extemely simple in 2d dv/dx - dv/dy
        vl = sample(vf, i - 1, j).y
        vr = sample(vf, i + 1, j).y
        vb = sample(vf, i, j - 1).x
        vt = sample(vf, i, j + 1).x
        vc = sample(vf, i, j)
        velocity_curls[i, j] = (vr - vl - vt+ vb) * scale


def step(mouse_data):
    #update the advect -> velocity and the dye
    # then projection to make sure the divergence free
    # update u_next tmp
    advect(velocities_pair.cur, velocities_pair.cur, velocities_pair.nxt,_intermedia_velocities)
    advect(velocities_pair.cur, dyes_pair.cur, dyes_pair.nxt, _intermedia_dye_buffer)
    velocities_pair.swap()
    dyes_pair.swap()
    
    apply_impulse(velocities_pair.cur, dyes_pair.cur, mouse_data)
    divergence(velocities_pair.cur)
    
    if conjugate_gradient:
        cgg.init(velocity_divs, -1)
        cgg.solve(max_iters=16)
        cgg.get_result(pressures_pair.cur)
    else:
        for _ in range(p_jacobi_iters):
            pressure_jacobi(pressures_pair.cur,pressures_pair.nxt)
            pressures_pair.swap()
    subtract_gradient(velocities_pair.cur, pressures_pair.cur)
    
#directly use the taichi example's code
class MouseDataGen(object):
    def __init__(self):
        self.prev_mouse = None
        self.prev_color = None

    def __call__(self, gui):   #instance could be called via this method () very similar to init but after initialization
        #[0:2] normalized delta direction
        #[2:4] current mouse xy
        #[4:7] color - rgb form
        mouse_data = np.zeros(8, dtype = np.float32)
        if gui.is_pressed(ti.GUI.LMB):
            mxy = np.array(gui.get_cursor_pos(), dtype = np.float32)*res
            if self.prev_mouse is None:
                self.prev_mouse = mxy
                # Set lower bound to 0.3 to prevent too dar colors
                self.prev_color = (np.random.rand(3)*0.7)+0.3
            else:
                mdir = mxy - self.prev_mouse
                mdir = mdir / (np.linalg.norm(mdir) + 1e-5)
                mouse_data[0], mouse_data[1] = mdir[0], mdir[1]
                mouse_data[2], mouse_data[3] = mxy[0], mxy[1]
                mouse_data[4:7] = self.prev_color
                self.prev_mouse = mxy
        else:
            self.prev_mouse = None
            self.prev_color = None
        
        return mouse_data

def reset():
    velocities_pair.fill(0)
    pressures_pair.fill(0)
    dyes_pair.fill(0)

gui = ti.GUI('Interactive Fluid', (res, res))
mouse_datagen = MouseDataGen()
while gui.running:
    if gui.get_event(ti.GUI.PRESS):
        e = gui.event
        if e.key == ti.GUI.ESCAPE:
            break
        elif e.key == 'r':
            reset()
    
    mouse_data = mouse_datagen(gui)
    step(mouse_data)
    gui.set_image(dyes_pair.cur)
    gui.show()



