OFFSET = [10,10]

NUM_POINTS = 50

# which dim is linear
DIM_INDEX = 1

# which axis is surface normal to the trapezoid of the wing
TRAPZ_IDX = 0

# needs to flip xy dir? - doesnt change a thing? -> have to *-1 the y values
DIM_FLIP_X = True
DIM_FLIP_Y = False

# number of segments number of different curves used for interpolation
NUM_SEGMENTS = 20

INPUT_NAME = "rear-wing-test-for-cnc-cutter.stl"
# INPUT_NAME = "Allerion_NO_horn.stl"
OUTPUT_NAME = "out.ngc"

# epsilon for points on same axis
# EPS = 2.220446049250313e-16
EPS = 1

# for parallel line check in trapezoid
PARALLEL_EPS = 0.001

# Hot wire length (mm)
HOTWIRE_LENGTH = 300

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