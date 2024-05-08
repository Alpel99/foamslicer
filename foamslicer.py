import meshio, datetime, itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.spatial import ConvexHull
from config import DIM_INDEX, DIM_FLIP_X, DIM_FLIP_Y , NUM_POINTS, OFFSET, NUM_SEGMENTS, OUTPUT_NAME, HOTWIRE_LENGTH, GCODE_INIT, INPUT_NAME, EPS, PARALLEL_EPS, TRAPZ_IDX

np.set_printoptions(suppress=True)

def getExtremePoints(val, mesh, idx):
    data = []
    for p in mesh:
        if abs(p[idx] - val) < EPS:
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
    # return np.sum(abs(points[1:]-points[:-1]))
    return np.sum(np.sqrt(np.abs(points[1:, 0]-points[:-1, 0])))
        
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
    if(s < 1e-15): return np.ones((num_segments,)).astype(int)
    data = (data/s)*num_points
    indices = [data < 1]
    off = np.sum(indices)
    # data[tuple(indices)] = 1
    return data.astype(int)

def evenOutPPs(pps1, pps2):
    if (d := np.sum(pps1)) != NUM_POINTS:
            r = np.argsort(pps1) if d < NUM_POINTS else np.argsort(pps1)[::-1]
            for i in range(abs(d-NUM_POINTS)):
                ind = r[i % len(pps1)]
                pps1[ind] = pps1[ind] +1 if d < NUM_POINTS else pps1[ind] -1 if pps1[ind] > 0 else pps1[ind]
                    
    if (s1 := np.sum(pps1)) != (s2 := np.sum(pps2)):
        r = np.argsort(pps2) if d < NUM_POINTS else np.argsort(pps2)[::-1]
        for i in range(abs(d-NUM_POINTS)):
            ind = r[i % len(pps2)]
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
        file.write("G1 X%.2f Y%.2f A%.2f Z%.2f\n" % (X, Y, U, V))
        
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
    minv = np.min(mesh, axis=0)
    shifted_mesh = np.array(mesh) - minv
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
        maxc1 = np.max(c1, axis=0)
        maxc2 = np.max(c2, axis=0)
        c1 = c1*-1 + maxc1
        c2 = c2*-1 + maxc2
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


def rotation_matrix(axis, angle):
    axis = axis / np.linalg.norm(axis)
    
    u_x, u_y, u_z = axis
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    one_minus_cos = 1 - cos_theta
    
    R11 = cos_theta + u_x**2 * one_minus_cos
    R12 = u_x * u_y * one_minus_cos - u_z * sin_theta
    R13 = u_x * u_z * one_minus_cos + u_y * sin_theta
    R21 = u_y * u_x * one_minus_cos + u_z * sin_theta
    R22 = cos_theta + u_y**2 * one_minus_cos
    R23 = u_y * u_z * one_minus_cos - u_x * sin_theta
    R31 = u_z * u_x * one_minus_cos - u_y * sin_theta
    R32 = u_z * u_y * one_minus_cos + u_x * sin_theta
    R33 = cos_theta + u_z**2 * one_minus_cos
        
    rotation_matrix = np.array([[R11, R12, R13],
                                [R21, R22, R23],
                                [R31, R32, R33]])
    
    return rotation_matrix

def rotateMesh(points, parallelLinePoints, axis, pos):
    p1, p2 = parallelLinePoints[0]
    slope = (p2[1] - p1[1])/(p2[0]-p1[0])
    rotation_angle = -np.arctan(slope) if not pos else np.arctan(slope)
    # print("rotation_angle (radians)", rotation_angle)
    
    rotation_axis = np.zeros((3,))
    rotation_axis[axis] = 1
    
    rot_mat = rotation_matrix(rotation_axis, rotation_angle)
    rotated_mesh = np.dot(points, rot_mat)
    return rotated_mesh

def find_trapezoid_corners(points):
    # probably faster with sorting first?
    min_x = np.min(points[:, 0])
    max_x = np.max(points[:, 0])
    min_y = np.min(points[:, 1])
    max_y = np.max(points[:, 1])
    
    min_x_points = points[points[:, 0] == min_x]
    max_x_points = points[points[:, 0] == max_x]
    min_y_points = points[points[:, 1] == min_y]
    max_y_points = points[points[:, 1] == max_y]
  
    min_x_point = min_x_points[0] if len(min_x_points) == 1 else np.average(min_x_points, axis=0)
    min_y_point = min_y_points[0] if len(min_y_points) == 1 else np.average(min_y_points, axis=0)
    max_y_point = max_y_points[0] if len(max_y_points) == 1 else np.average(max_y_points, axis=0)
    max_x_point = max_x_points[0] if len(max_x_points) == 1 else np.average(max_x_points, axis=0)
    
    return np.array([min_x_point, max_x_point, min_y_point, max_y_point])


def find_parallel_pairs(points):
    # Initialize list to store parallel pairs
    parallel_pairs = []
    
    # Generate all combinations of pairs of points
    point_combinations = itertools.combinations(points, 2)
    
    # Iterate over each combination of point pairs
    for pair1, pair2 in itertools.combinations(point_combinations, 2):
        # Extract coordinates of the points
        x1, y1 = pair1[0]
        x2, y2 = pair1[1]
        x3, y3 = pair2[0]
        x4, y4 = pair2[1]
        
        # Find slopes of the lines formed by the pairs
        slope1 = (y2 - y1) / (x2 - x1)
        slope2 = (y4 - y3) / (x4 - x3)
        # Check if slopes are equal (parallel lines)
        if abs(abs(slope1) - abs(slope2)) < PARALLEL_EPS:
            parallel_pairs.append((pair1, pair2))
    
    if(len(parallel_pairs) > 0):
        return parallel_pairs[0]
    else:
        raise Exception("No parallel pairs in trapezoid view found, is the shape off?")


def flipMesh(mesh, flipy, flipx, dim_idx):
    flips = [flipy, flipx]
    fliplist = [flips[i%2] if i != dim_idx else False for i in range(3)]
    
    flipped_mesh = mesh.copy()
    for i,f in enumerate(fliplist):
        if f:
            flipped_mesh[:, i] = -flipped_mesh[:, i]
    return flipped_mesh

def getSplines(points, nsegments):
    data = [points[0][0]]
    xmax = np.argmax(points[:, 0])
    xmin = np.argmin(points[:, 0])
    l = len(points)
    # print(l, xmax, xmin)
    # print(points[0], points[l-1], points[xmax], points[xmin])
    lens = np.zeros((l-1,))
    for i in range(l-1):
        lens[i] = getLength(points[i:i+2])
    lensum = np.sum(lens)
    lenperseg = lensum/nsegments
    splines = [None]*nsegments
    flip, flip2 = False, False
    c0,c = 0,0
    for i in range(0, nsegments):
        if(c < l-1):
            segsum = 0
            while(segsum < lenperseg and c < l-1):
                segsum += lens[c]
                c += 1
                ## only have splines with good xvalues
                if(c != 0 and c != l and (c == xmax or c == xmin)): 
                    print("Break")
                    data.append(points[c][0])
                    flip = not flip
                    break
            # print(c0, c)
            x = points[c0:c+1, 0] if flip2 else points[c0:c+1, 0][::-1]
            y = points[c0:c+1, 1] if flip2 else points[c0:c+1, 1][::-1]
            flip2 = flip
            # print(x)
            cs = CubicSpline(x, y)
            c0 = c
            splines[i] = (cs, np.sum(lens[c0:c]))
    return lensum, splines


def getPointsFromSplines(lensum, splines, num_points):
    dPoints = lensum/num_points
    data = np.zeros((num_points, 2), dtype=float)
    sidx = 0
    trlsum = 0
    for i in range(num_points):
        d = dPoints*i
        while(splines[sidx][1] < d):
            trlsum += splines[sidx][1]
            sidx += 1
        data[i][0] = dPoints*i
        data[i][1] = splines[sidx][0]()

    return data
    
if __name__ == "__main__":
    mesh = meshio.read(INPUT_NAME)

    # only work with points as mesh
    points = mesh.points
    hull = ConvexHull(points)
    indxs = [[[0,1],2], [[1,2],0], [[0,2],1]]
    hullpoints = points[hull.vertices][:, indxs[TRAPZ_IDX][0]]
    maxPoints = find_trapezoid_corners(hullpoints)
    # plotPoints(maxPoints, True)
    parallelpair = find_parallel_pairs(maxPoints)

    # plotPoints(points[:, indxs[0][0]], True)
    # plotPoints(points[:, indxs[1][0]], True)
    # plotPoints(points[:, indxs[2][0]], True)    

    # rotated_mesh = points
    rotated_mesh = rotateMesh(points, parallelpair, indxs[TRAPZ_IDX][1], True)
    # plotPoints(rotated_mesh[:, indxs[0][0]], True)
    # plotPoints(rotated_mesh[:, indxs[1][0]], True)
    # plotPoints(rotated_mesh[:, indxs[2][0]], True)

    flipped_mesh = flipMesh(rotated_mesh, DIM_FLIP_Y, DIM_FLIP_X, DIM_INDEX)


    shifted_mesh = shiftMesh(flipped_mesh)
    
    maxmin = getMeshMaxMin(shifted_mesh)
    np.savetxt("mesh.txt", shifted_mesh, fmt="%f")
    # print(maxmin)
    c1 = getOrderedExtremePoints(maxmin, shifted_mesh, 0)
    c2 = getOrderedExtremePoints(maxmin, shifted_mesh, 1)
    # c1 = c1*1.5
    # c2 = c2+[10,1]

    print("c1", len(c1))
    print("c2", len(c2))

    plotPoints(c1, show=True)
    splines1 = getSplines(c1, NUM_SEGMENTS)
    print(splines1)


    cp1 = getPointsFromSplines(splines1, NUM_POINTS)

    exit()
    
    # c1,c2 = flipPoints(c1, c2, DIM_FLIP_Y, DIM_FLIP_X)
    
    num_segments, segment_indices = getSegments(c1, c2)

    pps1 = getPointsPerSegment(c1, NUM_POINTS, num_segments, segment_indices)
    pps2 = getPointsPerSegment(c2, NUM_POINTS, num_segments, segment_indices)

    pps1, pps2 = evenOutPPs(pps1, pps2)
    pps2 = pps1
    c1p = getEvenPoints(c1, pps1, num_segments)
    c2p = getEvenPoints(c2, pps2, num_segments)
    
    c2pe = getExtendedPoints(c2p, c1p, maxmin[0][DIM_INDEX], maxmin[1][DIM_INDEX])
    
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

    
