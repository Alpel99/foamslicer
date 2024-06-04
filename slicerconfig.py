import config as text_config

class Foamconfig():
    def __init__(self):
        self.getConfig()

    def getConfig(self):
        self.input_file = text_config.INPUT_FILE
        self.offset = text_config.OFFSET
        self.num_points = text_config.NUM_POINTS
        self.dim_index = text_config.DIM_INDEX
        self.trapz_idx = text_config.TRAPZ_IDX
        self.dim_flip_x = text_config.DIM_FLIP_X
        self.dim_flip_y = text_config.DIM_FLIP_Y
        self.dim_flip_z = text_config.DIM_FLIP_Z
        self.num_segments = text_config.NUM_SEGMENTS
        self.output_name = text_config.OUTPUT_NAME
        self.eps = text_config.EPS
        self.parallel_eps = text_config.PARALLEL_EPS
        self.x_eps = text_config.X_EPS
        self.hotwire_length = text_config.HOTWIRE_LENGTH
        self.hotwire_offset = text_config.HOTWIRE_OFFSET
        self.hotwire_width = text_config.HOTWIRE_WIDTH
        self.gcode_init = text_config.GCODE_INIT
        self.workpiece_size = text_config.WORKPIECE_SIZE
        self.gcode_axis = text_config.GCODE_AXIS
        self.gcode_g1 = text_config.GCODE_G1

    def reset(self):
        self.getConfig()

    def writeConfig(self):
        print("Missing impl, setup writing of file")
        pass