import taichi as ti
import taichi.math as tm
import random
from datetime import datetime 
import time


ti.init(arch=ti.gpu)
#for bitonic sort, num of array must be power of 2

N_arr =  8;   #8192*128;
ti_arr = ti.field(dtype=ti.i32, shape=(N_arr, ));

@ti.kernel
def generate_array_taichi():
    for i in range(N_arr):
        ti_arr[i] = ti.random(ti.i32)%20000;
    #print("Generated Values: ", end=' ')
    #for i in range(N_arr):
    #    print(ti_arr[i], end=' ');
    #print("");

@ti.kernel
def print_array_taichi():
    print("Sorted Values: ", end=' ')
    for i in range(N_arr):
        print(ti_arr[i], end=' ');
    print("");
            

def generate_array(N):
    random.seed(datetime.now().timestamp());
    arr = []
    for i in range(N):
        arr.append(random.randint(0, 20000));   
    return arr;

def compAndSwap(a, i, j, dir):
    if (dir == 1 and a[i] > a[j]) or (dir == 0 and a[i]< a[j]):
        a[i], a[j] = a[j], a[i]

#recursion method 
def bitonicMerge(a, low, cnt, dir):
    if cnt >1:
        k = cnt//2;
        for i in range(low, low+k):
            compAndSwap(a, i, i+k, dir);
        bitonicMerge(a, low, k, dir);
        bitonicMerge(a, low+k, k, dir);   #merge less
 
def bitonicSort_CPU(a, low, cnt, dir):
    if cnt >1:
        k = cnt//2;   #floor division
        bitonicSort_CPU(a, low, k, 1); #increase
        bitonicSort_CPU(a, low+k, k, 0);  #descend
        bitonicMerge(a, low, cnt, dir);


#iteration method, a big step (k) to sort and a small step (j) to merger
def bitonicSort_CPU_iter(a, low, num, dir):
    k = 2;
    while k <= num:  # for sort
        j= (k//2);
        while j >0:  # for merge
            iter = num//j;
            low = 0;
            for i in range(iter-1):   # for swap
                for id in range(low, low+j):
                    compAndSwap(a, id, id+j, dir);
                low +=j;
            j=j//2;
        k*=2;   

@ti.func
def ti_swap(i, j):
    temp = ti_arr[i];
    ti_arr[i] = ti_arr[j];
    ti_arr[j] = temp;

@ti.kernel
def bitonic_sort_step(j: int, k: int, dir: int):
    for i in range(ti_arr.shape[0]):
        ixj = i^j;
        if ixj >i:   #dir control
            if (i&k) ==0:
                if  (ti_arr[i] > ti_arr[ixj]) and dir==1:   #ascending
                    ti_swap(i, ixj);
                elif (ti_arr[i] <=ti_arr[ixj]) and dir == 0:
                    ti_swap(i, ixj);
            else:
                if (ti_arr[i] <ti_arr[ixj]) and dir == 1:  #descending
                    ti_swap(i,ixj);
                elif (ti_arr[i] >=ti_arr[ixj]) and dir == 0:
                    ti_swap(i, ixj);
 
def bitonicSort_GPU(dir: int):
    k = 2;
    while k <=N_arr:
        j= (k//2);
        while j >0:
            bitonic_sort_step(j, k, dir);
            j=j//2;
        k*=2;                  
def main():
    
    start_time = time.time();
    
    #cpu way
    arr = generate_array(8);    
    n = len(arr);
    dir = 0;   # 1 - ascending, 0 - descending 
    #bitonicSort_CPU(arr, 0, n, dir);
    bitonicSort_CPU_iter(arr, 0, n, dir);
    for i in range(n):
        print("%d "%arr[i], end= " ");
    
    #gpu way
    #generate_array_taichi();
    #bitonicSort_GPU(0);
    #print_array_taichi();
    
    print("------time cost: %.6f s" %(time.time() - start_time));
    
    
if __name__ == "__main__":
    main();