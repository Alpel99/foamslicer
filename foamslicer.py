import meshio, ezdxf
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import slicerconfig as slicerconfig
import main as helpers
import numpy as np


class Foamslicer():
    def __init__(self):
        self.config = slicerconfig.Foamconfig()
        self.alignidxs = [[[0,1],2], [[1,2],0], [[0,2],1]]
        self.c1 = self.c2 = None
        self.cp1 = self.cp2 = None
        self.cp1e = self.cp2e = None
        self.c1old = self.c2old = None
        self.dxf = False

    def generate3DPoints(self):
        self.applyShapeOffset()
        cp1edim = -self.config.hotwire_offset
        cp2edim = self.config.hotwire_length - self.config.hotwire_offset
        cp1dim = 0
        self.points3d = np.zeros((3,))
        if(self.dxf):
            cp2dim = self.config.workpiece_size
        else:
            cp2dim = self.maxmin[1][self.config.dim_index]
        
        if self.cp1e is not None:
            self.cp1e3d = np.insert(self.cp1e, 2, cp1edim, axis=1)
            self.points3d = np.vstack((self.points3d, self.cp1e3d))
        if self.cp2e is not None:
            self.cp2e3d = np.insert(self.cp2e, 2, cp2edim, axis=1)
            self.points3d = np.vstack((self.points3d, self.cp2e3d))
        if self.cp1 is not None:
            self.cp13d = np.insert(self.cp1, 2, cp1dim, axis=1)
            self.points3d = np.vstack((self.points3d, self.cp13d))
        if self.cp2 is not None:
            self.cp23d = np.insert(self.cp2, 2, cp2dim, axis=1)
            self.points3d = np.vstack((self.points3d, self.cp23d))
        # print(self.cp1e[0], self.cp1[0])

    def alignMesh(self):
        self.points = helpers.alignMesh(self.points, self.config.dim_index, self.config.mode)

    def curveNormalization(self):
        if self.c1old is None:
            self.c1old = self.c1.copy()
            self.c2old = self.c2.copy()

        # interpolation with offset factor
        l1 = helpers.getLength(self.c1)
        l2 = helpers.getLength(self.c2)
        if(l2 > l1):
            l1,l2 = l2,l1
        deltaL = l1-l2
        avgL = (l1+l2)/2
        y = self.config.hotwire_width_factor * deltaL/avgL + 1
        
        f1 = 1 if l1 > l2 else y
        f2 = 1 if l2 > l1 else y
        # print(f1*self.config.hotwire_width, f2*self.config.hotwire_width)
        self.c1 = helpers.padCurve(self.c1old, f1*self.config.hotwire_width)
        self.c2 = helpers.padCurve(self.c2old, f2*self.config.hotwire_width)

    def writeGCode(self):
        c = self.config
        if(self.cp1e is not None):
            res = helpers.writeFile(self.cp1e, self.cp2e, c.offset, c.gcode_init, c.gcode_axis, c.gcode_g1)
        elif(self.cp1 is not None):
            res = helpers.writeFile(self.cp1, self.cp2, c.offset, c.gcode_init, c.gcode_axis, c.gcode_g1)
        elif(self.c1 is not None):
            res = helpers.writeFile(self.c1, self.c1, c.offset, c.gcode_init, c.gcode_axis, c.gcode_g1)
        else:
            return False
        return res

    def applyShapeOffset(self):
        self.shape_offset = helpers.getOffset(self.c1, self.c2, self.cp1, self.cp2, self.cp1e, self.cp2e)
        if(self.c1 is not None):
            self.c1 -= self.shape_offset
            self.c2 -= self.shape_offset
        if(self.cp1 is not None):
            self.cp1 -= self.shape_offset
            self.cp2 -= self.shape_offset
        if(self.cp1e is not None):
            self.cp1e -= self.shape_offset
            self.cp2e -= self.shape_offset
            self.true_offset = -self.shape_offset + self.config.offset

    def getExtendedPoints(self):
        if(not self.dxf):
            m1 = self.maxmin[0][self.config.dim_index]
            m2 = self.maxmin[1][self.config.dim_index]
        else:
            m1 = 0
            m2 = self.config.workpiece_size
        offset = self.config.hotwire_length - self.config.hotwire_offset
        helpers.checkHotwireDim(m1, m2)
        if(self.config.workpiece_size == self.config.hotwire_length):
            self.cp2e = self.cp2.copy()
        else:
            self.cp2e = helpers.getExtendedPoints(self.cp2, self.cp1, m2, m1, offset)

        if(self.config.hotwire_offset > 0):
            self.cp1e = helpers.getExtendedPoints(self.cp1, self.cp2, m1, m2, -self.config.hotwire_offset)
        else:
            self.cp1e = self.cp1.copy()

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
        if self.dxf:
            if(c.dim_flip_x):
                self.c1[:, 0] = -self.c1[:, 0]
                self.c2[:, 0] = -self.c2[:, 0]
            if(c.dim_flip_y):
                self.c1, self.c2 = self.c2, self.c1
                if self.cp1 is not None:
                    self.cp1, self.cp2 = self.cp2, self.cp1
                if self.cp1e is not None:
                    self.cp1e, self.cp2e = self.cp2e, self.cp1e
                self.c1old = None
            if(c.dim_flip_z):
                self.c1[:, 1] = -self.c1[:, 1]
                self.c2[:, 1] = -self.c2[:, 1]
            self.applyShapeOffset()
        else:    
            dim_idx = self.config.dim_index
            flips = [c.dim_flip_x, c.dim_flip_y, c.dim_flip_z]
            self.points = helpers.flipMesh(self.points, flips, dim_idx)

    def alignMeshAxis(self):
        self.hull = ConvexHull(self.points)
        align_idx = self.alignidxs[self.config.trapz_idx]
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
        dist = 0.1
        segs = max(int(self.config.num_segments/4), 2)
        d1 = ezdxf.readfile(p1)
        self.c1 = helpers.readDXF(d1, dist, segs)
        if(p2):
            d2 = ezdxf.readfile(p2)
            self.c2 = helpers.readDXF(d2, dist, segs)
        else:
            self.c2 = self.c1.copy()
        self.applyShapeOffset()
        self.c1 = helpers.orderPoints(self.c1)
        self.c2 = helpers.orderPoints(self.c2)

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
            self.dxf = False
        elif end == "dxf":
            self.dxf = True
            if not arr_sw:
                self.readDXF(path)
            if arr_sw and len(input) == 1:
                self.readDXF(input[0])    
            if arr_sw and len(input) == 2:
                self.readDXF(input[0], input[1])
        else:
            raise("No fitting file found, use stl, 1 dxf or 2 dxf")


if __name__ == "__main__":
    slicer = Foamslicer()
    slicer.config.input_file="files/rear-wing-test-for-cnc-cutter.stl"
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