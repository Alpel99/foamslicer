import meshio
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import slicerconfig as slicerconfig
import main as helpers


class Foamslicer():
    def __init__(self):
        self.config = slicerconfig.Foamconfig()
        self.alignidxs = [[[0,1],2], [[1,2],0], [[0,2],1]]

    def alignMesh(self):
        self.points = helpers.alignMesh(self.points, self.config.dim_index, self.config.mode)

    def curveNormalization(self):
        self.c1old = self.c1
        self.c2old = self.c2
        self.c1 = helpers.extendPoints(self.c1old, self.config.hotwire_width)
        self.c2 = helpers.extendPoints(self.c2old, self.config.hotwire_width)

    def writeGCode(self):
        c = self.config
        res = helpers.writeFile(self.cp1e, self.cp2e, c.offset, c.gcode_init)
        return res

    def applyShapeOffset(self):
        self.shape_offset = helpers.getOffset(self.cp1e, self.cp2e)
        self.cp1e -= self.shape_offset
        self.cp2e -= self.shape_offset
        self.true_offset = -self.shape_offset + self.config.offset

    def getExtendedPoints(self):
        m1 = self.maxmin[0][self.config.dim_index]
        m2 = self.maxmin[1][self.config.dim_index]
        offset = self.config.hotwire_length - self.config.hotwire_offset
        self.cp2e = helpers.getExtendedPoints(self.cp2, self.cp1, m1, m2, offset)
        if(self.config.hotwire_offset > 0):
            self.cp1e = helpers.getExtendedPoints(self.cp1, self.cp2, m1, m2, self.config.hotwire_offset)
        else:
            self.cp1e = self.cp1

    def getPointsFromSplines(self):
        c = self.config
        self.cp1 = helpers.getPointsFromSplines(self.lensum1, self.splines1, c.num_points)
        self.cp2 = helpers.getPointsFromSplines(self.lensum2, self.splines2, c.num_points)

    def getSplines(self):
        c = self.config
        self.lensum1, self.splines1 = helpers.getSplines(self.c1, c.num_segments)
        self.lensum2, self.splines2 = helpers.getSplines(self.c2, c.num_segments)

    def getOrderedExtremePoints(self):
        c = self.config
        self.c1 = helpers.getOrderedExtremePoints(self.maxmin, self.points, 0, c.dim_index, c.x_eps)
        self.c2 = helpers.getOrderedExtremePoints(self.maxmin, self.points, 1, c.dim_index, c.x_eps)

    def getMeshMaxMin(self):
        self.maxmin = helpers.getMeshMaxMin(self.points)

    def shiftMesh(self):
        self.points = helpers.shiftMesh(self.points)

    def flipMesh(self):
        c = self.config
        dim_idx = self.config.dim_index
        flips = [c.dim_flip_x, c.dim_flip_y, c.dim_flip_z]
        self.points = helpers.flipMesh(self.points, flips, dim_idx)

    def alignMeshAxis(self):
        self.hull = ConvexHull(self.points)
        align_idx = self.alignidxs[self.config.trapz_index]
        hullpoints = self.points[self.hull.vertices][:, align_idx[0]]
        maxPoints = helpers.find_trapezoid_corners(hullpoints)
        parallelPair = helpers.find_parallel_pairs(maxPoints)
        self.points = helpers.rotateMesh(self.points, parallelPair, align_idx[1], True)

    def getPoints(self):
        self.points = self.mesh.points
        return self.points
    
    def readSTL(self, path):
        self.mesh = meshio.read(path)

    def readDXF(self, p1, p2=None):
        pass

    def readFiles(self):
        input = self.config.input_file
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
    slicer = Foamslicer()
    slicer.readFiles()
    points = slicer.getPoints()
    # print(points)
    # helpers.create3dplot(points)
    # slicer.flipMesh()
    # slicer.alignMeshAxis()
    # helpers.create3dplot(slicer.points)
    # exit()
    slicer.alignMeshAxis()
    slicer.flipMesh()
    slicer.shiftMesh()
    slicer.getMeshMaxMin()
    slicer.getOrderedExtremePoints()
    slicer.curveNormalization()
    slicer.getSplines()
    slicer.getPointsFromSplines()
    slicer.getExtendedPoints()
    slicer.applyShapeOffset()
    helpers.plotPoints(slicer.cp1e)
    helpers.plotPoints(slicer.cp2e, True)
    slicer.writeGCode()