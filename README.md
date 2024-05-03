# foamslicer
gcode generator from stl files for airfoils

# pre-data
stl from random thingyverse
https://www.thingiverse.com/thing:94917/files

gcode from online airfoil gcode generator
https://www.diyrcwings.com/app/

# Documentation
* c1: big side, where foam starts
* c2: potential smaller side, potential space to foam
    * can swap/flip with flipx, flipy


# ideas/todo
## extension to hotwire length does not work at all
* maybe try with 3d shear matrix, this could work imo

## rotation of initial stl could be off:
* kinda works for 1 given axis with trapezoid now
* maxbe should try to align/rotate other axis then aswell

* get extreme points
    * basically trapezoid
* align axis with one of the sides
    * need option to align with other sides
        * e.g. dunno which one might be smaller


## pre points ?
* need some stuff to flip?
* some sort of length offset between piece and real dimension
    * shrink one side to account for it
        * interpolate lines to accomodate for length distr?

## gcode
* set init stuff, whatever -> heat wire etc

* other problems:
* in what coordinates does gcode work? - mm
    * conversion of some sort? - assume mm for stl?
* standard offset from 0,0? -> foam not directly at start
    * maybe add standard entering offset to certainly be in foam

* different speeds?
    * simply coordinates for both shapes, nothing changes
    * steps small enough for smaller one to not stop ?
        * increase overall speed, if too long at place & smelting


## basic 3d visualization of stl
```python
# import open3d as o3d

mesh = o3d.io.read_triangle_mesh("Allerion_NO_horn.stl")
mesh = mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh], window_name="STL", left=1000, top=200, width=800, height=650)
```