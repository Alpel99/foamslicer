import numpy as np
import ezdxf

def print_entity(e):
    print("LINE on layer: %s" % e.dxf.layer)
    print("start point: %s" % e.dxf.start)
    print("end point: %s" % e.dxf.end)

doc = ezdxf.readfile("diamond.dxf")
msp = doc.modelspace()

# points = np.empty()
points = []
for e in msp:
    if e.dxftype() == "LINE":
        points.append(e.dxf.start)
        points.append(e.dxf.end)
    if e.dxftype() == "POLYLINE":
        # parts of polyline could be arc aswell?
        for v in e.dxf.points():
            points.append(v)
    if e.dxftype() == "ARC":
        print("blyaat")
        c = e.dxf.center
        r = e.dxf.radius
        a1 = e.dxf.start_angle
        a2 = e.dxf.end_angle
        # maybe! e.dxf.flattening() works

points = np.array(points)
print(points)

from main import create3dplot
create3dplot(points)