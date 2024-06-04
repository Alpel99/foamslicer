import numpy as np
from config import NUM_SEGMENTS, NUM_POINTS
import ezdxf

from main import readDXF, plotPoints


dist = 0.05
segments = 4

d1 = ezdxf.readfile("files/wing-left-part4-V1_inside.DXF")
d2 = ezdxf.readfile("files/wing-left-part4-V1_outside.DXF")
p1 = readDXF(d1, dist, segments)
p2 = readDXF(d2, dist, segments)
# print(points)

# print(p1, p2)
# create3dplot(p1)
plotPoints(p1)
plotPoints(p2, True)