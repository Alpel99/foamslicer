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

print(maxmin)

maxmin[0] = np.max(arr, axis=0)
maxmin[1] = np.min(arr, axis=0)

print("new\n", maxmin)

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

# print("d1", d1)

x_values = [point[0] for point in d2]
y_values = [point[1] for point in d2]

# plt.plot(x_values, y_values, 'bo')  # 'bo' means blue circles

print("d1", d1[:, 1])

for i in range(0, len(d1[:]), 1):
    print(i)
    plt.plot(d1[i:i+2, 0], d1[i:i+2, 1], 'bo-')

plt.show()

# how to generate gcode?

def generatePath(points):
    pass
