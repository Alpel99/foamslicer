# import open3d as o3d

# mesh = o3d.io.read_triangle_mesh("Allerion_NO_horn.stl")
# mesh = mesh.compute_vertex_normals()
# o3d.visualization.draw_geometries([mesh], window_name="STL", left=1000, top=200, width=800, height=650)

import meshio
import numpy as np
import matplotlib.pyplot as plt

mesh = meshio.read("Allerion_NO_horn.stl")

arr = np.array(mesh.points)

maxmin = np.zeros((2,3))
maxmin[0] = np.max(arr, axis=0)
maxmin[1] = np.min(arr, axis=0)

# print("new\n", maxmin)

def getExtremePoints(val, idx):
    data = []
    for p in mesh.points:
        if abs(p[idx] - val) < 2.220446049250313e-16:
            pn = []
            for j in range(0, len(p)):
                if(j != idx):
                    pn.append(p[j])
            data.append(pn)
    return np.array(data)

i = 1
d1 = getExtremePoints(maxmin[0][i], i)
d2 = getExtremePoints(maxmin[1][i], i)

# plt.plot(x_values, y_values, 'bo')  # 'bo' means blue circles

def plotPoints(d1):
    plt.plot(d1[:, 0], d1[:, 1], 'bo-')
    # p = [-3.55537148,  2.44643784]
    # plt.plot(p[0], p[1], "ro")
    plt.show()

# how to generate gcode?

def orderPoints(points):
    barycenter = np.mean(points, axis=0)
    print(barycenter)
    vectors = points - barycenter
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    sorted_indices = np.argsort(angles)
    ordered = points[sorted_indices]
    ordered = np.vstack([ordered, ordered[0]])
    return ordered
    
convex = orderPoints(d1)
# plotPoints(convex)

def getSpeedCoeff(p1, p2):
    pass

def getOffset(p1, p2):
    pass


