# foamslicer
gcode generator from stl or dxf files for airfoils (or other convex shapes)

![screenshot](https://i.ibb.co/sRXWZrp/Screenshot-from-2024-06-04-13-53-31.png)

# ideas/todo
* rotations could be done differently:
    * take min/max/middle axis of dim
    * rotate this to align with axis [this works kind of decently rn]
    * problem with middle -> how to find, need to be already orth on 1 axis
* probably lots of bugs still
* catch more errors?
* curve padding is dependent on order of points
    * is currently fine tho
* input for curve padding
* curve padding reset does not work or smthg

## todo Documentation
* write proper documentation
    * one for users: kinda done
    * one for devs/in depth
        * what do the functions do/idea behind them


# User Documentation
## Setup
```bash
git clone https://github.com/Alpel99/foamslicer.git
pip install -r requirements.txt
python3 slicergui.py
```

## Initial Configuration
* (should be) constant values set in `config.py`
* comments to explain shortly what is going on

- OFFSET: motor offset from 0,0 to move into foam
    - in mm
    - should be fine with no weird interactions
- NUM_POINTS: number of points from splines, set in gui
- DIM_INDEX: dimension of extreme points without gui
- TRAPZ_IDX: only for rotation without gui
- DIM_FLIP*: for flips without gui
- NUM_SEGMENTS: (aka NumSplines):
    - how many splines to interpolate the wing shape
- INPUT_FILE: set input file, can be done in gui
- OUTPUT_NAME: name of written gcode file
- EPS: to determine points from stl to be in plane 
    - max diff of values from max
- PARALLEL_EPS: for rotation/alignment
- X_EPS: to determine/group maximum points
- HOTWIRE_LENGTH: length between the two motor planes
- HOTWIRE_OFFSET: length from first motor to workpiece in mm
- HOTWIRE_WIDTH: width of the wire/how much is melted in mm
    - used by curve padding
- WORKPIECE_SIZE: used if dxf files
    - needs dimension of workpiece for proper offsets
- GCODE_INIT: what commands should the GCODE start with
    - from [diyrcwings](https://www.diyrcwings.com/app/)
- GCODE_AXIS: names for the gcode axis (might differ)
    - grbl is weird, we use XY, ZA
- GCODE_G1: whether to use G0 or G1 commands as movement
- HOTWIRE_WIDTH_FACTOR: factor determined by ratio of cut lengths
    - experimental


## Walkthrough
* start with `python3 slicergui.py`
* should open file if given one in config
    * else file explorer
        * choose 1 stl, or 1/2 dxf files
* radiobuttons for axis selection are used by multiple buttons below (flip, rotate, extremePoints)
* plot will stack (except 3d/2d stuff)
    * can empty with button
* open new file at the bottom (all progress lost)
* some buttons greyed out
    * because data/variables do not exist yet
    * need to use others first
* finally: write GCode file

### STL Usage
* rotate the mesh to align with any axis (use radio button axis selection)
    * still wip, works sometimes decently
    * can dynamically rotate in 3d mesh to get an idea what is going on
* flip mesh around selected axis
* get extreme points on that selected axis
    * might need to flip around this axis aswell
        * extension only works 1 way (wip)

### DXF Usage
* dxf files only need 2d plots
* no need for rotation/extreme points
* can still flip around selected axis
* need to specify workpiece width
    * to calculate extension properly

### For Both
* curve padding to add constant on curve normal
* numPoints and numSegments to generate splines and get even points
* extendedPoints to calculate the offsets
    * basically extend shape to motor movement plane
    * distance for this needs to be set with dxf files
        * cannot be given in dxf file, is in stl
* generate GCODE


# Dev Documentation
* want to give some explanation/train of thought for functions in this
## main.py/helpers functions
### readDXF
* ezdxf library
* iterate through modelspace
* check for dxftypes
* can currently parse
    * LINE
    * POLYLINE
    * SPLINE
* need to add more, if not enough
* throws warning which dxftypes are not recognized

### read stl
* uses meshio
* pretty straight forward
* get points from mesh as numpy 3d array

### alignMeshAxis
* function in foamslicer.py
* uses multiple functions and a bit of magic
* take convex hull of points (scipy)
* trying to align some 2d view (top, left, front side)
    * towards one axis (most outer points parallel to it)
* `find_trapezoid_corners` on hullpoints to find points with max x/y
    * points that should be aligned
* `find_parallel_pairs` orders these maxpoints and tries to find 2 parallel lines to align with
* `rotateMesh` uses a rotation matrix on the entire mesh
    * matrix is 3d, but one row+column is 001, so doesn't change a thing
* approach/routine could need some improvement overall

### flipMesh
* flips points (*-1) and applies new offset for geometry to lie in > 0,0

### getOrderedExtremePoints
* have aligned mesh and values of max plane
    * get those points
* order the extreme points by calculating `atan2` from its xmin point
    * ensures that the top part is first instead of the bottom
    * could use some barycenter/mean aswell, xmin is easy and works

### getSplines
* need plines to generate evenly distributed points on the airfoil shape
    * need those points to cut extend the points properly to motor axis
    * helps for even gcode mvt
* need unique points for splines -> filter xvalues
* parameterize splines by distance -> cumulative sum of lengths
* iterate over points, keep track of length of current segment, next spline if too long
* use scipy `make_interp_spline`
    * once for x and once for y
* apparently also works very well with just 1 spline for the entire airfoil 

### getPointsFromSplines
* iterate previously generate splines
* keep track of distance to get even points

### getExtendedPoints
* extend points to motor planes to cut the proper shape out of foam
* get proper dimensions of workpiece/cutter in config
* iterate through all point pairs
* calculate line through both, get value of it at required distance

### writeGCode
* writes gcode file
* one function reused to write G1/G0 lines
* iterate all points and put proper values
* quite a bit of config stuff (G1 vs G0, axis names)
    * so lots of parameters

### curveNormalization/padCurve/curve padding
* has bad names
* is needed because hotwire smelts its way -> has width
    * makes final result smaller than input lines
* takes line between point left and right of current one
* normal of this line with length of offset value is added
* can yield shitty results at trailing edge
* currently uses parameter `HOTWIRE_WIDTH_FACTOR`
    * parameter to multiply the ratio of both lengths
    * pads the 2nd (shorter) one with more (times this factor)

### generate3DPoints
* in slicer
* takes even and extended points
* puts them at proper distances in 3d plot
* used to check if extension calculations make sense

### variables
* c1: input points
* cp1: even points (generated from c1)
* cp1old: even points, used to not stack curve padding
* cp1e: extended points -> motor plane movements
    * these should be used to write gcode

# License
MIT