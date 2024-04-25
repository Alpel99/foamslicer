# foamslicer
gcode generator from stl files for airfoils

# pre-data
stl from random thingyverse
https://www.thingiverse.com/thing:94917/files

gcode from online airfoil gcode generator
https://www.diyrcwings.com/app/

# ideas/todo
## pre points ?
* need some stuff to flip?
* some sort of length offset between piece and real dimension
    * shrink one side to account for it

## gcode
* set init stuff, whatever -> heat wire etc
* standard speed: move offset
* with speedcoeff: move both through their geometries
    * maybe need 2 coeffs: one for top, one for bottom
    * need to rech reverse point at the same time

* other problems:
* in what coordinates does gcode work?
    * conversion of some sort?
* standard offset from 0,0? -> foam not directly at start
    * maybe add standard entering offset to certainly be in foam