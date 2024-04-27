# import open3d as o3d

# mesh = o3d.io.read_triangle_mesh("Allerion_NO_horn.stl")
# mesh = mesh.compute_vertex_normals()
# o3d.visualization.draw_geometries([mesh], window_name="STL", left=1000, top=200, width=800, height=650)

import meshio, datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from config import DIM_INDEX, DIM_FLIP, NUM_POINTS, OFFSET, NUM_SEGMENTS, OUTPUT_NAME

np.set_printoptions(suppress=True)

def getExtremePoints(val, idx):
    data = []
    for p in shifted_mesh:
        if abs(p[idx] - val) < 2.220446049250313e-16:
            pn = []
            for j in range(0, len(p)):
                if(j != idx):
                    pn.append(p[j])
            data.append(pn)
    return np.array(data)

def plotPoints(points, show=False):
    plt.plot(points[:, 0], points[:, 1], 'o-')
    # p = [-3.55537148,  2.44643784]
    # plt.plot(p[0], p[1], "ro")
    if show: plt.show()

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
    diff = m1-m2
    return diff

def splitCurve(points):
    max_x_index = np.argmax(points[:, 0])
    arr1 = points[:max_x_index + 1]
    arr2 = points[max_x_index:]
    return arr1, arr2

def getEvenPoints(points, num_points, num_segments):
    # print("points", points)
    l = len(points[:, 0])
    num_segments = max(min(num_segments, l-1),1)
    segment_indices = np.linspace(0, l, num_segments + 1, dtype=int)
    diff = segment_indices[1:]-segment_indices[:-1]
    points_per_segment = int(num_points/num_segments)
    data = np.empty((0,2))
    # print(num_segments, l, segment_indices)
    for i in range(num_segments):
        # offset = 0 if i == 0 else 1
        # ind = points[segment_indices[i]-offset:segment_indices[i+1], :]
        ind = points[segment_indices[i]:segment_indices[i + 1] + 1, :]
        # print("ind", ind)
        cs = CubicSpline(np.arange(len(ind[:, 0])), ind)
        t = np.linspace(0, len(ind[:, 0])-1, points_per_segment)
        data = np.vstack([data, cs(t)])
    return data

def writeG1Lines(file, points1, points2):
    for p1, p2 in zip(points1, points2):
        file.write(f"G1 X{p1[0]} Y{p1[1]} U{p2[0]} V{p2[1]}\n")
        


if __name__ == "__main__":
    mesh = meshio.read("Allerion_NO_horn.stl")

    minv = np.min(mesh.points, axis=0)

    shifted_mesh = np.array(mesh.points) + abs(minv)
    # print(shifted_mesh)

    maxmin = np.zeros((2,3))
    maxmin[0] = np.max(shifted_mesh, axis=0)
    maxmin[1] = np.min(shifted_mesh, axis=0)
    # print(maxmin)
    
    d1 = getExtremePoints(maxmin[0][DIM_INDEX], DIM_INDEX)
    d2 = getExtremePoints(maxmin[1][DIM_INDEX], DIM_INDEX)
    d2 = d2*1.5
    d1 = d1+[10,1]
    # plotPoints(d1)
    # plotPoints(d2)
    # plt.show()

    #convex1
    c1 = orderPoints(d1)
    c2 = orderPoints(d2)
    # plotPoints(c1)
    # plotPoints(c2)
    # plt.show()
    
    c1t, c1b = splitCurve(c1)
    c2t, c2b = splitCurve(c2)

    c1tp = getEvenPoints(c1t, NUM_POINTS, NUM_SEGMENTS)
    c1bp = getEvenPoints(c1b, NUM_POINTS, NUM_SEGMENTS)
    c2tp = getEvenPoints(c2t, NUM_POINTS, NUM_SEGMENTS)
    c2bp = getEvenPoints(c2b, NUM_POINTS, NUM_SEGMENTS)
    
    plotPoints(c1tp)   
    plotPoints(c1bp)
    plotPoints(c2tp)   
    plotPoints(c2bp)
    plt.show()
    
    # d1 x times faster than d2
    speedcoefft = getLength(c1tp)/getLength(c1bp)
    speedcoeffb = getLength(c2tp)/getLength(c2bp)
    print(speedcoefft, speedcoeffb)
    
    offset = getOffset(c1, c2)
    print(offset)
    
    file1 = open(OUTPUT_NAME, "w")
    file1.write(f"( foamslicer.py, at {datetime.datetime.now()} )\n")
    file1.close()
    file1 = open(OUTPUT_NAME, "a")
    writeG1Lines(file1, c1tp, c2tp)
    
