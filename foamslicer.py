import meshio, datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from config import DIM_INDEX, DIM_FLIP_X, DIM_FLIP_Y , NUM_POINTS, OFFSET, NUM_SEGMENTS, OUTPUT_NAME, HOTWIRE_LENGTH, GCODE_INIT

np.set_printoptions(suppress=True)

def getExtremePoints(val, mesh, idx):
    data = []
    for p in mesh:
        if abs(p[idx] - val) < 2.220446049250313e-16:
            pn = []
            for j in range(0, len(p)):
                if(j != idx):
                    pn.append(p[j])
            data.append(pn)
    return np.array(data)

def plotPoints(points, show=False, name="", lbl=""):
    plt.plot(points[:, 0], points[:, 1], 'o-', label=lbl)
    # p = [-3.55537148,  2.44643784]
    # plt.plot(p[0], p[1], "ro")
    plt.axis('equal')
    plt.legend()
    if show:
        plt.show()
    if not show and len(name) > 0:
        plt.savefig("figures/" + name)

def orderPoints(points):
    barycenter = np.mean(points, axis=0)
    # print(barycenter)
    vectors = points - barycenter
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    sorted_indices = np.argsort(angles)
    ordered = points[sorted_indices]
    ordered = np.vstack([ordered, ordered[0]])
    return ordered

def getLength(points):
    return np.sum(abs(points[1:]-points[:-1]))
        
def getOffset(points1, points2):
    m1 = np.min(points1, axis=0)
    m2 = np.min(points2, axis=0)
    diff = m2-m1
    return diff

def splitCurve(points):
    max_x_index = np.argmax(points[:, 0])
    arr1 = points[:max_x_index + 1]
    arr2 = points[max_x_index:]
    return arr1, arr2

def getPointsPerSegment(points, num_points, num_segments, segment_indices):
    data = np.zeros((num_segments,))
    for i in range(num_segments):
        l = 0
        for j in range(segment_indices[i], segment_indices[i+1]):
            l += getLength(points[j:j+2])
        data[i] = l
    s = np.sum(data)
    data = (data/s)*num_points
    return data.astype(int)

def evenOutPPs(pps1, pps2):
    if (d := np.sum(pps1)) != NUM_POINTS:
            r = np.argsort(pps1) if d < NUM_POINTS else np.argsort(pps1)[::-1]
            for i in range(abs(d-NUM_POINTS)):
                if i == len(pps1): i = 0
                ind = r[i]
                pps1[ind] = pps1[ind] +1 if d < NUM_POINTS else pps1[ind] -1 if pps1[ind] > 0 else pps1[ind]
                    
    if (s1 := np.sum(pps1)) != (s2 := np.sum(pps2)):
        r = np.argsort(pps2) if d < NUM_POINTS else np.argsort(pps2)[::-1]
        for i in range(abs(d-NUM_POINTS)):
            if i == len(pps2): i = 0
            ind = r[i]
            pps2[ind] = pps2[ind] +1 if s2 < s1 else pps2[ind] -1 if pps2[ind] > 0 else pps2[ind]        
    return pps1, pps2

def getEvenPoints(points, points_per_segment, num_segments):
    data = np.empty((0,2))
    # print(num_segments, l, segment_indices)
    for i in range(num_segments):
        # offset = 0 if i == 0 else 1
        # ind = points[segment_indices[i]-offset:segment_indices[i+1], :]
        ind = points[segment_indices[i]:segment_indices[i + 1] + 1, :]
        # print("ind", ind)
        cs = CubicSpline(np.arange(len(ind[:, 0])), ind)

        t = np.linspace(0, len(ind[:, 0])-1, points_per_segment[i])
        data = np.vstack([data, cs(t)])
    return data

def getExtendedPoints(p1, p2, p1x, p2x):
    data = np.zeros(p1.shape)
    t = (p2x - p1x)/HOTWIRE_LENGTH
    for i, (pp1, pp2) in enumerate(zip(p1, p2)):
        data[i] = pp2 + t * (pp1 - pp2)
    return data

def writeGcodeInit(file, gcode_init):
    file.write(gcode_init)
    
def writeG1Lines(file, points1, points2):
    for p1, p2 in zip(points1, points2):
        X = round(p1[0], 2)
        Y = round(p1[1], 2)
        U = round(p2[0], 2)
        V = round(p2[1], 2)
        file.write("G1 X%.2f Y%.2f U%.2f V%.2f\n" % (X, Y, U, V))
        
def writeOffsetMvt(file, shape_offset, offset):
    file.write(("( OFFSET )\n"))
    writeG1Lines(file, [offset], [offset])
    file.write(("( OFFSET + SHAPE OFFSET )\n"))
    writeG1Lines(file, [offset], [offset+shape_offset])
    
def reverseOffsetMvt(file, shape_offset, offset):
    file.write(("( BACK TO INIT POS )\n"))
    writeG1Lines(file, [offset], [offset+shape_offset])
    file.write(("( REVERSE SHAPE OFFSET )\n"))
    writeG1Lines(file, [offset], [offset])
    file.write(("( REVERSE OFFSET / GO TO ZERO )\n"))
    writeG1Lines(file, [[0,0]], [[0,0]])     
    
def shiftMesh(mesh):
    minv = np.min(mesh.points, axis=0)
    shifted_mesh = np.array(mesh.points) + abs(minv)
    return shifted_mesh
        
def checkHotwireDim(maxmin):
    cut_d = maxmin[0][DIM_INDEX] - maxmin[1][DIM_INDEX]
    # print("cut_d", cut_d)
    d_diff = HOTWIRE_LENGTH - cut_d
    if(d_diff) < 0:
        raise Exception("Distance in stl greater than HOTWIRE_LENGTH")
    
def getMeshMaxMin(mesh):
    maxmin = np.zeros((2,3))
    maxmin[0] = np.max(mesh, axis=0)
    maxmin[1] = np.min(mesh, axis=0)
    return maxmin

def getOrderedExtremePoints(maxmin, mesh, idx):
    points = getExtremePoints(maxmin[idx][DIM_INDEX], mesh, DIM_INDEX)
    # plotPoints(d1)
    #convex
    return orderPoints(points)

def flipPoints(c1, c2, flipy, flipx):
    if(flipx):
        c1, c2 = c2, c1
    if(flipy):
        c1 *= -1
        c2 *= -1
    return c1, c2

def getSegments(c1, c2):
    # set proper num_segments
    l = min(len(c1[:, 0]), len(c2[:, 0]))
    num_segments = max(min(NUM_SEGMENTS, l-1),1)
    segment_indices = np.linspace(0, l, num_segments + 1, dtype=int)
    return num_segments, segment_indices

def writeFile(c1p, c2p, shape_offset):
    file1 = open(OUTPUT_NAME, "w")
    file1.write(f"( foamslicer.py, at {datetime.datetime.now()} )\n")
    file1.close()
    file1 = open(OUTPUT_NAME, "a")
    writeGcodeInit(file1, GCODE_INIT)
    writeOffsetMvt(file1, shape_offset, OFFSET)
    file1.write(("( SHAPE )\n"))
    writeG1Lines(file1, c1p, c2p)
    reverseOffsetMvt(file1, shape_offset, OFFSET)

if __name__ == "__main__":
    mesh = meshio.read("Allerion_NO_horn.stl")
    shifted_mesh = shiftMesh(mesh)
    
    maxmin = getMeshMaxMin(shifted_mesh)

    c1 = getOrderedExtremePoints(maxmin, shifted_mesh, 0)
    c2 = getOrderedExtremePoints(maxmin, shifted_mesh, 1)
    c1 = c1*1.5
    c2 = c2+[10,1]
    
    c1,c2 = flipPoints(c1, c2, DIM_FLIP_Y, DIM_FLIP_X)
    
    num_segments, segment_indices = getSegments(c1, c2)

    pps1 = getPointsPerSegment(c1, NUM_POINTS, num_segments, segment_indices)
    pps2 = getPointsPerSegment(c2, NUM_POINTS, num_segments, segment_indices)

    pps1, pps2 = evenOutPPs(pps1, pps2)
    
    c1p = getEvenPoints(c1, pps1, num_segments)
    c2p = getEvenPoints(c2, pps2, num_segments)
    
    c2pe = getExtendedPoints(c1p, c2p, maxmin[0][DIM_INDEX], maxmin[1][DIM_INDEX])
    
    plotPoints(c1p, lbl="c1p")
    plotPoints(c2p, lbl="c2p")
    plotPoints(c2pe, True, lbl="c2pe")

    shape_offset = getOffset(c1p, c2pe)
    # print("shape_offset", shape_offset)
    # print("c1p", c1p)
    # print("c2p", c2p)
    # print("c2pe", c2pe)
    
    true_offset = shape_offset + OFFSET
    writeFile(c1p + true_offset, c2pe + true_offset, shape_offset)

    
