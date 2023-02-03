#only support cpu and cuda backend for now
import taichi as ti 

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
#screen rendering init
Width, Height = 800, 600;
resolution = (Width, Height);
grid_size = 8;
iTime = ti.field(float, shape=());

#create edge tables for 16 cases
edge_table = ti.Matrix.field(2, 2, int, 16);

vec2 = ti.math.vec2;
Edge = ti.types.struct(p0=vec2, p1=vec2);   #declare struct type
edges = Edge.field();   #declare variable
D_Node = ti.root.dynamic(ti.i, 1024, chunk_size=32);
D_Node.place(edges);

#assume unit length - 1, and square's range from [-10, 10]



if __name__ == "__main__":
    add_data();
    clear_data();