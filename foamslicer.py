# import open3d as o3d

# mesh = o3d.io.read_triangle_mesh("Allerion_NO_horn.stl")
# mesh = mesh.compute_vertex_normals()
# o3d.visualization.draw_geometries([mesh], window_name="STL", left=1000, top=200, width=800, height=650)

import meshio
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)

### INPUTS
# which dim is linear
i = 1
# needs to flip xy dir?
flip = False

mesh = meshio.read("Allerion_NO_horn.stl")

min = np.min(mesh.points, axis=0)

shifted_mesh = np.array(mesh.points) + abs(min)
# print(shifted_mesh)

maxmin = np.zeros((2,3))
maxmin[0] = np.max(shifted_mesh, axis=0)
maxmin[1] = np.min(shifted_mesh, axis=0)
# print(maxmin)

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


d1 = getExtremePoints(maxmin[0][i], i)
d2 = getExtremePoints(maxmin[1][i], i)
d2 = d2*1.5
d1 = d1+[10,1]

def plotPoints(points, show=False):
    plt.plot(points[:, 0], points[:, 1], 'o-')
    # p = [-3.55537148,  2.44643784]
    # plt.plot(p[0], p[1], "ro")
    if show: plt.show()

# plotPoints(d1)
# plotPoints(d2)
# plt.show()

def orderPoints(points):
    barycenter = np.mean(points, axis=0)
    # print(barycenter)
    vectors = points - barycenter
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    sorted_indices = np.argsort(angles)
    ordered = points[sorted_indices]
    ordered = np.vstack([ordered, ordered[0]])
    return ordered
    
#convex1
c1 = orderPoints(d1)
c2 = orderPoints(d2)
plotPoints(c1)
plotPoints(c2)
plt.show()

def getLength(points):
    return np.sum(abs(points[1:]-points[:-1]))
        
# d1 x times faster than d2
speedcoeff = getLength(c1)/getLength(c2)
# print(speedcoeff)

def getOffset(points1, points2):
    m1 = np.min(points1, axis=0)
    m2 = np.min(points2, axis=0)
    diff = m1-m2
    print(diff)

getOffset(c1, c2)

