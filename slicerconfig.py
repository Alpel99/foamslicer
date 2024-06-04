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
        attributes = [a for a in dir(self) if not a.startswith('__') and not callable(getattr(self, a))]
        data = []
        switch = False
        with open("config.py") as file:
            for line in file:
                if "'''" in line:
                    switch = not switch
                attr = line.split('=')[0].strip().lower()
                if attr in attributes:
                    var = getattr(self, attr)
                    if type(var) is not str and type(var) is not list:
                        data.append(f"{attr.upper()} = {var}\n")
                    else:
                        if type(var) is list:
                            if type(var[0]) is not str:
                                data.append(f"{attr.upper()} = {var}\n")
                            else:
                                data.append(f"{attr.upper()} = [")
                                data.append(",".join([f"'{x}'" for x in var]))
                                data.append("]\n")
                        
                        if type(var) is str:
                            if('\n' in var):
                                data.append(f"{attr.upper()} = '''{var}'''\n")
                            else:
                                data.append(f"{attr.upper()} = '{var}'\n")
                else:
                    if not switch and "'''" not in line:
                        data.append(line)

        with open('config.py', 'w') as file:
            for line in data:
                file.write(line)