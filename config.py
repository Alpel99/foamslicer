OFFSET = [10,10]

NUM_POINTS = 100

# which dim is linear
DIM_INDEX = 1

# which axis is surface normal to the trapezoid of the wing
TRAPZ_IDX = 0

# needs to flip xy dir? - doesnt change a thing? -> have to *-1 the y values
DIM_FLIP_X = False
DIM_FLIP_Y = False
DIM_FLIP_Z = True

# number of segments number of different curves used for interpolation
NUM_SEGMENTS = 40

INPUT_NAME = "rear-wing-test-for-cnc-cutter.stl"
# INPUT_NAME = "Allerion_NO_horn.stl"
OUTPUT_NAME = "out_gui.ngc"

# epsilon for points on same axis
# EPS = 2.220446049250313e-16
EPS = 1

# for parallel line check in trapezoid
PARALLEL_EPS = 0.001

# epsilon for minimum x, in order to have 1 final point, and no problems with overlapping edges
X_EPS = 0.1

# Hot wire length (mm)
HOTWIRE_LENGTH = 300

# Offset from 0 point (motor movement plane) (mm)
HOTWIRE_OFFSET = 0

# Wire width
HOTWIRE_WIDTH = 1.2


GCODE_INIT = '''G17
G21
( SET ABSOLUTE MODE )
G90
( SET CUTTER COMPENSATION )
G40
( SET TOOL LENGTH OFFSET )
G49
( SET PATH CONTROL MODE )
G64
( SET FEED RATE MODE )
G94
F300
'''