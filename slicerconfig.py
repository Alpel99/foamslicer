import json, os

class Foamconfig():
    def __init__(self):
        self.set_default_config()
        self.ensure_config_file("config.json")
        self.getConfig()

    def getConfig(self):
        self.loadJson()
        config = self.jsonconfig
        self.offset = config.get('OFFSET', self.default_config["OFFSET"])
        self.num_points = config.get('NUM_POINTS', self.default_config["NUM_POINTS"])
        self.dim_index = config.get('DIM_INDEX', self.default_config["DIM_INDEX"])
        self.trapz_idx = config.get('TRAPZ_IDX', self.default_config["TRAPZ_IDX"])
        self.dim_flip_x = config.get('DIM_FLIP_X', self.default_config["DIM_FLIP_X"])
        self.dim_flip_y = config.get('DIM_FLIP_Y', self.default_config["DIM_FLIP_Y"])
        self.dim_flip_z = config.get('DIM_FLIP_Z', self.default_config["DIM_FLIP_Z"])
        self.num_segments = config.get('NUM_SEGMENTS', self.default_config["NUM_SEGMENTS"])
        self.input_file = config.get('INPUT_FILE', self.default_config["INPUT_FILE"])
        self.output_name = config.get('OUTPUT_NAME', self.default_config["OUTPUT_NAME"])
        self.eps = config.get('EPS', self.default_config["EPS"])
        self.parallel_eps = config.get('PARALLEL_EPS', self.default_config["PARALLEL_EPS"])
        self.x_eps = config.get('X_EPS', self.default_config["X_EPS"])
        self.hotwire_length = config.get('HOTWIRE_LENGTH', self.default_config["HOTWIRE_LENGTH"])
        self.hotwire_offset = config.get('HOTWIRE_OFFSET', self.default_config["HOTWIRE_OFFSET"])
        self.hotwire_width = config.get('HOTWIRE_WIDTH', self.default_config["HOTWIRE_WIDTH"])
        self.workpiece_size = config.get('WORKPIECE_SIZE', self.default_config["WORKPIECE_SIZE"])
        self.gcode_init = config.get('GCODE_INIT', self.default_config["GCODE_INIT"])
        self.gcode_axis = config.get('GCODE_AXIS', self.default_config["GCODE_AXIS"])
        self.gcode_g1 = config.get('GCODE_G1', self.default_config["GCODE_G1"])
        self.hotwire_width_factor = config.get('HOTWIRE_WIDTH_FACTOR', self.default_config["HOTWIRE_WIDTH_FACTOR"])

    def reset(self):
        self.getConfig()

    def writeConfig(self):
        config = {
            "OFFSET": self.offset,
            "NUM_POINTS": self.num_points,
            "DIM_INDEX": self.dim_index,
            "TRAPZ_IDX": self.trapz_idx,
            "DIM_FLIP_X": self.dim_flip_x,
            "DIM_FLIP_Y": self.dim_flip_y,
            "DIM_FLIP_Z": self.dim_flip_z,
            "NUM_SEGMENTS": self.num_segments,
            "INPUT_FILE": self.input_file,
            "OUTPUT_NAME": self.output_name,
            "EPS": self.eps,
            "PARALLEL_EPS": self.parallel_eps,
            "X_EPS": self.x_eps,
            "HOTWIRE_LENGTH": self.hotwire_length,
            "HOTWIRE_OFFSET": self.hotwire_offset,
            "HOTWIRE_WIDTH": self.hotwire_width,
            "WORKPIECE_SIZE": self.workpiece_size,
            "GCODE_INIT": self.gcode_init,
            "GCODE_AXIS": self.gcode_axis,
            "GCODE_G1": self.gcode_g1,
            "HOTWIRE_WIDTH_FACTOR": self.hotwire_width_factor
        }
          
        with open("config.json", 'w') as file:
            json.dump(config, file, indent=4)

    def ensure_config_file(self, file_path):
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                json.dump(self.default_config, f, indent=4)
            print(f"Configuration file created with default values at {file_path}")

    def loadJson(self):
        with open('config.json', 'r') as file:
            self.jsonconfig = json.load(file)

    def set_default_config(self):
        self.default_config = {
            "OFFSET": [10, 10],
            "NUM_POINTS": 500,
            "DIM_INDEX": 1,
            "TRAPZ_IDX": 1,
            "DIM_FLIP_X": False,
            "DIM_FLIP_Y": True,
            "DIM_FLIP_Z": False,
            "NUM_SEGMENTS": 1,
            "INPUT_FILE": [],
            "OUTPUT_NAME": "out.ngc",
            "EPS": 1,
            "PARALLEL_EPS": 0.001,
            "X_EPS": 0.1,
            "HOTWIRE_LENGTH": 1000,
            "HOTWIRE_OFFSET": 200,
            "HOTWIRE_WIDTH": 1.2,
            "WORKPIECE_SIZE": 600,
            "GCODE_INIT": "G17\nG21\n( SET ABSOLUTE MODE )\nG90\n( SET CUTTER COMPENSATION )\nG40\n( SET TOOL LENGTH OFFSET )\nG49\n( SET PATH CONTROL MODE )\nG64\n( SET FEED RATE MODE )\nG94\nF300\n",
            "GCODE_AXIS": ['X','Y','A','Z'],
            "GCODE_G1": True,
            "HOTWIRE_WIDTH_FACTOR": 2.18
        }


if __name__ == '__main__':
    c = Foamconfig()
    print(c.gcode_init)
    c.writeConfig()