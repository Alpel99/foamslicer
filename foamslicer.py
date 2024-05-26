import meshio, datetime, itertools, click
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, make_interp_spline
from scipy.spatial import ConvexHull
import config as text_config
import main as helpers


class foamslicer():
    def __init__(self,):
        self.config = {}
        self.initConfig()
        self.alignidxs = [[[0,1],2], [[1,2],0], [[0,2],1]]

    def alignMeshAxis(self):
        self.getHull(self.points)
        align_idx = self.alignidxs[self.config["trapz_idx"]]
        hullpoints = self.points[self.hull.vertices][:, align_idx[0]]
        maxPoints = helpers.find_trapezoid_corners(hullpoints)
        parallelPair = helpers.find_parallel_pairs(maxPoints)
        self.mesh = helpers.rotateMesh(self.points, parallelPair, align_idx[1])

    def getHull(self):
        self.hull = ConvexHull(self.points)
        return self.hull

    def getPoints(self):
        self.points = self.mesh.points
        return self.points

    def initConfig(self):
        c = self.config
        c["input_file"] = text_config.INPUT_NAME
        c["offset"] = text_config.OFFSET
        c["num_points"] = text_config.NUM_POINTS
        c["dim_index"] = text_config.DIM_INDEX
        c["trapz_index"] = text_config.TRAPZ_IDX
        c["dim_flip_x"] = text_config.DIM_FLIP_X
        c["dim_flip_y"] = text_config.DIM_FLIP_Y
        c["num_segments"] = text_config.NUM_SEGMENTS
        c["output_name"] = text_config.OUTPUT_NAME
        c["eps"] = text_config.EPS
        c["parallel_eps"] = text_config.PARALLEL_EPS
        c["x_eps"] = text_config.X_EPS
        c["hotwire_length"] = text_config.HOTWIRE_LENGTH
        c["hotwire_offset"] = text_config.HOTWIRE_OFFSET
        c["gcode_init"] = text_config.GCODE_INIT
    
    def readSTL(self, path):
        self.mesh = meshio.read(path)

    def readDXF(self, p1, p2=None):
        pass

    def readFiles(self):
        input = self.config["input_file"]
        arr_sw = False
        if(type(input) == str):
            end = input.split(".")[-1].lower()
            path = input
        if(type(input) == list):
            arr_sw = True
            path = input[0]
            end = input[0].split(".")[-1].lower()
        if end == "stl":
            self.readSTL(path)
            return
        if end == "dxf":
            if not arr_sw:
                self.readDXF(path)
                return
            if arr_sw and len(input) == 2:
                self.readDXF(input[0], input[1])
                return
        raise("No fitting file found, use stl, 1 dxf or 2 dxf")
    



if __name__ == "__main__":
    slicer = foamslicer()
    slicer.readFiles()
    slicer.getPoints()
    print(slicer.points)