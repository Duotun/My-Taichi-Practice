#only support cpu and cuda backend for now
#OIT - Sequence-unrelated transparency rendering
import taichi as ti
import taichi.math as tm 

ti.init(arch=ti.cuda)

#Screen Show Init
res = (1000, 1000);
color_buffer = ti.Vector.field(3, dtype=ti.f32, shape=res);

eps = 1e-4;
inf = 1e10;
fov = 0.5;
aspect_ratio = res[0] / res[1];

#Scene Data
alpha_min = 0.2;
alpha_width = 0.3;
cam_pos = tm.vec3(0., 0., 5.);
background_color = tm.vec4(0.2, 0.2, 0.2, 1.);

Line = ti.types.struct(pos=tm.vec3, dir=tm.vec3);
Sphere = ti.types.struct(center=tm.vec3, radius=float, color=tm.vec4);
spheres = Sphere.field();   # spheres as 3d object to construct the scene
ti.root.dynamic(ti.i, 1024, chunk_size=64).place(spheres);

ColorWithDepth = ti.types.struct(color=tm.vec4, depth=float);
colors_in_pixel = ColorWithDepth.field();
ti.root.dense(ti.ij, res).dynamic(ti.k, 2048, chunk_size=64).place(colors_in_pixel);


#rendering utilize one-pass gooch NPR(no edge stuff)

@ti.func
def gooch_lighting(normal: ti.template()):
    light = tm.normalize(tm.vec3(-1., 2., 1.));
    warmth = normal*light*0.5 + 0.5;
    return tm.mix(tm.vec3(0., 0.25, 0.75), tm.vec3(1., 1., 1.), warmth);

#assign color to objects (spheres this case)
@ti.func
def shading(color, normal: ti.template()):
    inColor = color.rgb * gooch_lighting(normal);
    alpha = tm.clamp(alpha_min + color.a * alpha_width, 0., 1.);
    return tm.vec4(inColor, alpha);


#assume rays from the image coordinates (camera_pos, considered as well)
#assume image is one-unit in the front of camera
@ti.func  
def gen_cam_ray(u, v):
    #y-[-1, 1], x - [-aspect, aspect], z - related to the fov / focal_length
    ray_dir = tm.vec3((2 * (u+0.5)/res[0]-1)* aspect_ratio, 2 * (v+0.5)/res[1]-1., -1.0/fov);
    ray_dir = tm.normalize(ray_dir);
    return Line(pos=cam_pos, dir=ray_dir);

#intersection process
#line-sphere intersections
@ti.func 
def intersect_sphere(line: Line, sphere: ti.template()):
    colorA = tm.vec4(0.);
    colorB = tm.vec4(0.);
    dist_1 = inf;
    dist_2 = inf;
    dl = sphere.center - line.pos;
    dl_dis = dl.dot(dl);
    per_pendicular = dl.dot(line.dir);  #small means per_pendicular
    r_dis = sphere.radius * sphere.radius
    out_of_sphere = (dl_dis > r_dis);
    is_intersection = True;   #default is true

    #check perpendicular case / or reversed directions
    if -eps < dl_dis -r_dis <eps:   #if very close
        if -eps < per_pendicular < eps:
            is_intersection = False;
        out_of_sphere = (per_pendicular < 0.);   #reversed dir or not
    
    if per_pendicular < 0. and out_of_sphere:
        is_intersection = False 
    
    #check intersection case (two-potential points)
    if is_intersection == True:
        in_radius =  -dl_dis +r_dis + per_pendicular * per_pendicular;
        if in_radius >=0.0:
            in_radius = tm.sqrt(in_radius);
            t1 = per_pendicular -  in_radius;
            if t1>0:
                hit_pos1 = line.pos + line.dir *t1;
                dist_1 = t1;
                normal1 = tm.normalize(hit_pos1 - sphere.center);
                colorA = shading(sphere.color, normal1);
            t2 = per_pendicular + in_radius;
            if t2>0:
                hit_pos2 = line.pos + line.dir *t2;
                dist_2 = t2;
                normal2 = tm.normalize(hit_pos2 - sphere.center);
                colorB = shading(sphere.color, normal2);
    return ColorWithDepth(color=colorA, depth = dist_1), ColorWithDepth(color=colorB, depth=dist_2);

#calculate interseciton each pixel, expand the colors_in_pixel
@ti.func
def get_intersection_pixel(u,v):
    line = gen_cam_ray(u, v);
    colors_in_pixel[u, v].deactivate();
    for i in range(spheres.length()):
        hit1, hit2 = intersect_sphere(line, spheres[i]);
        if hit1.depth < inf:
            colors_in_pixel[u,v].append(hit1);
        if hit2.depth < inf:
            colors_in_pixel[u,v].append(hit2);

#sort the dynamic nodes in each pixel case (ascending order)
@ti.func 
def bubble_sort(u,v):
    plen = colors_in_pixel[u, v].length();              
    for i in range(plen-1):
        for j in range(plen-i-1):    # pop the largets to the last
            if colors_in_pixel[u,v,j].depth > colors_in_pixel[u, v, j+1].depth:
                tmp = colors_in_pixel[u,v,j].depth;
                colors_in_pixel[u,v,j].depth = colors_in_pixel[u,v,j+1].depth;
                colors_in_pixel[u,v,j+1].depth = tmp;

#blend color with depth sorted
#weighted with color.a
@ti.func
def blend(color, base_color):
    color.rgb += (1-color.a)*base_color.rgb * base_color.a 
    color.a += (1-color.a) * base_color.a;
    return color;

#blend from the pure black and transparent (ascending with depth)
@ti.func
def get_color(u,v):
     color = tm.vec4(0.0);
     for i in range(colors_in_pixel[u,v].length()):
         color = blend(color, colors_in_pixel[u, v, i].color);
     color = blend(color, background_color);
     gamma_corrected_color = tm.pow(color.rgb*color.a, 1./2.2);
     #print(color.a)
     color_buffer[u,v]= gamma_corrected_color;
    

#kernel stuff for parallel
@ti.kernel
def generate_spheres(n: ti.i32):
    spheres.deactivate();
    for i in range(n):
        spheres.append(Sphere(center=tm.vec3(ti.random()*3 - 1.5, ti.random()*3 - 1.5, ti.random()*3 - 1.5), radius=ti.random()*0.2+0.1,color=tm.vec4(ti.random(), ti.random(), ti.random(), ti.random())));

@ti.kernel
def render():
    for u,v in color_buffer:
        get_intersection_pixel(u,v);
        bubble_sort(u,v)
        get_color(u,v)

# simple gui to show the rendered image
def screen_show():
    gui = ti.GUI('OIT Rendering', res, fast_gui=True);
    generate_spheres(256);   #prepare the sphere scene
    render();
    gui.set_image(color_buffer);
    while gui.running:
        gui.show();

if __name__ == "__main__":
    screen_show();

