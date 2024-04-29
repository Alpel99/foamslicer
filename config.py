OFFSET = [10,10]

NUM_POINTS = 50

# which dim is linear
DIM_INDEX = 1

# needs to flip xy dir? - doesnt change a thing? -> have to *-1 the y values
DIM_FLIP_X = False
DIM_FLIP_Y = False

# number of segments number of different curves used for interpolation
NUM_SEGMENTS = 4

OUTPUT_NAME = "out.ngc"

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