#only support cpu and cuda backend for now
import taichi as ti 
import taichi.math as tm 
import numpy as np
import time  # for random generators


ti.init(arch=ti.cuda);

#Basic Test
S = ti.root.dense(ti.i,10).dynamic(ti.j, 1024, chunk_size=32);  #dynamic SNode as a tree-root structure
x = ti.field(int);   #define a variable, could add as a dense array holding lists
S.place(x);    #make it a dynamic list

@ti.kernel
def add_data():
    #ti.loop_config(serialize=True)
    for i in range(10):
        for j in range(i):
            x[i].append(j)
            print(x[i].length())

@ti.kernel
def clear_data():
    for i in range(10):
        x[i].deactivate()
        print(x[i].length())  #will print 0


#Marching Squares
#define local space coordinate (from interpolation)
#add grid positoin -> global space coordinate
#utilize Dynamic Node to form a complete update each frame in GPU of parallel ways

#screen rendering init
Width, Height = 800, 600;
resolution = (Width, Height);
grid_size = 8;
level = 0.15 # draw contour of isofunc = level
iTime = ti.field(float, shape=());
pixels = ti.Vector.field(3, float, shape = resolution);

#create edge tables for 18 cases  (2 cases are combined to solve ambiguity)
#becareful with the ambiguity (saddle point)
edge_table_np = np.array([   #follow the encode sequence (0 -15)
    [[-1, -1], [-1, -1]],
    [[3, 0], [-1, -1]],
    [[0, 1],[-1, -1]],
    [[1, 3], [-1, -1]],
    [[1, 2], [-1, -1]],
    [[0, 1], [2, 3]],
    [[0, 2], [-1, -1]],
    [[2, 3], [-1, -1]],
    [[2,3],[-1,-1]],
    [[0, 2], [-1, -1]],
    [[0, 3],[1,2]],
    [[1, 2],[-1, -1]],
    [[1, 3],[-1, -1]],
    [[0, 1], [-1, -1]],
    [[3, 0], [-1, -1]],  #14
    [[-1, -1],[-1,-1]]   #15
    ], dtype = np.int32);

edge_table = ti.Matrix.field(2, 2, int, 16);
edge_table.from_numpy(edge_table_np);

vec2 = ti.math.vec2;
Edge = ti.types.struct(p0=vec2, p1=vec2);   #declare struct type
edges = Edge.field();   #declare variable
D_Node = ti.root.dynamic(ti.i, 1024, chunk_size=32);
D_Node.place(edges);

#assume unit length - 1, and square's range from [-10, 10]

@ti.func
def hash22(p):
    n = tm.sin(tm.dot(p, tm.vec2(41, 289)))
    p = tm.fract(tm.vec2(262144, 32768) * n)
    return tm.sin(p * 6.28 + iTime[None])   # introduce time variable for random generator


@ti.func
def noise(p):
    ip = tm.floor(p)
    p -= ip
    v = tm.vec4(tm.dot(hash22(ip), p),
                tm.dot(hash22(ip + tm.vec2(1, 0)), p - tm.vec2(1, 0)),
                tm.dot(hash22(ip + tm.vec2(0, 1)), p - tm.vec2(0, 1)),
                tm.dot(hash22(ip + tm.vec2(1, 1)), p - tm.vec2(1, 1)))
    p = p * p * p * (p * (p * 6 - 15) + 10)
    return tm.mix(tm.mix(v.x, v.y, p.x), tm.mix(v.z, v.w, p.x), p.y)    #mix ->linear interpolation

@ti.func
def isofunc(p):
    return noise(p/4 +1);


#get the vertex position of the unit grid [[0, 1], [2, 3]] counter-clock 2D grid
#values refer to the isovalue on the grid, isovalue is the level (contour)
@ti.func
def get_vertex(vertex_id, values, isovalue):   #isovalu for the interpolatoin
    v = tm.vec2(0);
    square = [tm.vec2(0), tm.vec2(0, 1), tm.vec2(1, 1), tm.vec2(1, 0)];
    if vertex_id == 0:
        v = interp(square[0], square[1],values.x, values.y, isovalue);
    elif vertex_id == 1:
        v = interp(square[1], square[2],values.y, values.z, isovalue);
    elif vertex_id == 2:
        v = interp(square[2], square[3],values.z, values.w, isovalue);
    elif vertex_id == 3:
        v = interp(square[3], square[0],values.w, values.x, isovalue);
    return v;
    
#used for define the position of iso lines
@ti.func
def interp(p1, p2, v1, v2, interp_val):
    return tm.mix(p1, p2, (interp_val - v1)/(v2-v1));


#update marching_square with dynamic nodes each frame
#and in parallel way
@ti.kernel
def marching_square():
    edges.deactivate();    #clear the dynamic nodes
    x_range = tm.ceil(grid_size * Width/Height, int);
    y_range = grid_size;
    
    for i, j in ti.ndrange((-x_range, x_range),(-y_range, y_range)):
        t_id = 0;   # for accessing edge table
        grid_values = tm.vec4(isofunc(tm.vec2(i, j)), isofunc(tm.vec2(i, j+1)),
                              isofunc(tm.vec2(i+1, j+1)), isofunc(tm.vec2(i+1, j)));   #for coordinates
        # form the bit-mapping
        if grid_values.x > level:
            t_id |=1;
        if grid_values.y > level:
            t_id |=2;
        if grid_values.z > level:
            t_id |=4;
        if grid_values.w > level:
            t_id |=8;
        
        #Fix the ambiguity for case 5 and 10
        if t_id == 5 or t_id ==10:
            center_val = isofunc(tm.vec2(i+0.5, j+0.5));
            if center_val > level:
                t_id = 15 - t_id;
        
        #check whether to expand the edges
        for k in ti.static(range(2)):
            if edge_table[t_id][k, 0] != -1:
                ind_1 = edge_table[t_id][k, 0];
                ind_2 = edge_table[t_id][k, 1];
                p0 =  tm.vec2(i, j) + get_vertex(ind_1, grid_values, level);
                p1 =  tm.vec2(i, j) + get_vertex(ind_2, grid_values, level);
                edges.append(Edge(p0, p1));
        
#dseg -> calculate the vertical distance to the edge
@ti.func
def dseg(p, a, b):
    p -=a;
    b -=a;
    h = tm.clamp(tm.dot(p, b)/tm.dot(b, b), 0.0, 1.0);
    return tm.length(p - h*b);

#draw grid, edges, and iso areas
#follow the grid position to define the final color
#map the marching square to the renderer pages
@ti.kernel
def render():
    for i, j in pixels:
        p = (2 * tm.vec2(i, j) - tm.vec2(resolution))/Height;  #[-W/H, W/H] X [-1, 1]
        p *= grid_size
        #p.y +=1.2;
        # Background Color
        col = tm.vec3(0.3, 0.6, 0.8);  
        if isofunc(p) > level:    #iso - range, no countour lines interpolated
            col = tm.vec3(1, 0.8, 0.3)
            
        # Draw background grid (black lines), manhattan distance to the grid
        q = abs(tm.fract(p) - 0.5);
        dd = 0.5 - q;     
        dgrid = ti.min(dd.x, dd.y);
        col = tm.mix(col, tm.vec3(0), 1-tm.smoothstep(0,0.04, dgrid-0.02))  
        
        # Draw edges
        dedge = 1e5;
        dv = 1e5;
        
        for k in range(edges.length()):
            s, e = edges[k].p0, edges[k].p1;
            dedge = ti.min(dedge, dseg(p, s, e));  #close to the edges
            dv = ti.min(dv, tm.length(p-s), tm.length(p-e));
        
        col = tm.mix(col, tm.vec3(1, 0, 0),
                     1-tm.smoothstep(0, 0.05, dedge-0.02));
        
        # Draw small circles at the vertices
        # With boundary ways
        col = tm.mix(col, tm.vec3(0.), 1-tm.smoothstep(0, 0.05, dv - 0.1));
        col = tm.mix(col, tm.vec3(1.), 1-tm.smoothstep(0, 0.05, dv - 0.08));
         
        pixels[i, j] = tm.sqrt(tm.clamp(col, 0,1))   #gamma-correction

# For screen show
t0 = time.perf_counter();
gui = ti.GUI('Marching Square (2D)', res = resolution, fast_gui=True);    

def screen_show():
    while gui.running and not gui.get_event(gui.ESCAPE):
        iTime[None] = time.perf_counter() - t0;
        marching_square();
        render()
        gui.set_image(pixels);
        gui.show();

if __name__ == "__main__":
    #basic test
    #add_data();
    #clear_data();
    
    #marching square
    screen_show();