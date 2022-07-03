from numpy import dtype
import taichi as ti
from ray import Ray
import material

vec3f = ti.types.vector(3, ti.f32);
#hit_record also records object index 
hit_record = ti.types.struct(
    pos = vec3f, 
    normal=vec3f, 
    t = ti.f32,
    frontface = ti.i32,
)



@ti.func
def set_face_normal(ray, outward_normal):
    frontface = 1  if ray.direction.dot(outward_normal) <0.0 else 0;
    if frontface == 0:
        outward_normal = -1.0 * outward_normal;
    return frontface, outward_normal 

@ti.data_oriented
class hittable:
    def __init__(self):
        pass
    #use pass to represent an abstruct class
    @ti.func
    def hit(self, ray, t_min, t_max):
        pass

@ti.data_oriented
class Sphere (hittable):
    def __init__(self, s_center, s_radius, s_material):
        self.center = s_center;
        self.radius = s_radius;
        self.material = s_material;
        
    #return the hit information
    @ti.func
    def hit(self, ray, t_min, t_max):
        is_hit = False;
        rec = hit_record(pos = ti.Vector([0, 0, 0]), normal = ti.Vector([0, 0 ,0]), t = 0.0, frontface = 0);   # initialize directly, can't go with field
        oc = ray.origin - self.center;
        a = ray.direction.dot(ray.direction);
        half_b = oc.dot(ray.direction);
        c =  oc.dot(oc)  - self.radius * self.radius;
        discriminant = half_b* half_b - a*c;
        is_hit = False
        root = 0.0;
        if discriminant >= 0.0:
            sqrtd = ti.sqrt(discriminant);
            root = (-half_b - sqrtd)/a;
            #print("Root: ", root);
            if root < t_min or root > t_max:
                root = (-half_b + sqrtd)/a;
                if root >= t_min and root <=t_max:    # find the nearest t segments
                    is_hit = True;
            else:
                is_hit = True;
       
        #update the hit record information
        if is_hit:
            rec.t = root;
            rec.pos = ray.at(root);
            rec.frontface, rec.normal = set_face_normal(ray, (rec.pos - self.center) / self.radius);
        return is_hit, rec, self.material;


# follow the code implementations from taichi course
@ti.data_oriented
class hittable_list(hittable):
    def __init__(self):
        self.objects = []   # use list to contain all objects in the world
    def add(self, obj):
        self.objects.append(obj);
    def clear(self):
        self.objects = []; 
    
    @ti.func
    def get_obj(self, i):
        return self.objects[i];
    
    @ti.func
    def hit(self, ray, t_min=0.001, t_max=10e8):
        closest_t = t_max;
        is_hitanything = False;
        rec = hit_record(pos = ti.Vector([0, 0, 0]), normal = ti.Vector([0, 0 ,0]), t = 0.0, frontface = 0);
        mat = material._Material(color = vec3f(0.0, 0.0, 0.0), matindex =-1, roughness =0.0, ior = 0.0); 
        for index in ti.static(range(len(self.objects))):
            is_hittmp, rectmp, tmpmat = self.objects[index].hit(ray, t_min, closest_t);
            if is_hittmp:
                closest_t = rectmp.t;
                is_hitanything = is_hittmp;
                rec = rectmp;
                mat = tmpmat;
        return  is_hitanything, rec, mat;   #return the objIndex for the world indexes
