import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, make_interp_spline, UnivariateSpline
np.set_printoptions(suppress=True)

from foamslicer import getLength

points = np.genfromtxt("files/points.txt")
dists = np.genfromtxt("files/dists.txt")
# allpoints = np.genfromtxt("files/allpoints.txt")
# print(allpoints)
print(dists)
print(points)
# print(dists[0], dists[-1])
# print(points[0], points[-1])
plt.plot(points[:, 0], points[:, 1], "x-", label="orig", linewidth=4)

# cs = CubicSpline(dists, points)
cs = make_interp_spline(dists, points, k=1)

weights = (dists[1:]-dists[:-1])
weights = np.concatenate(([weights[0]],weights))
print(weights)

t = dists / np.sum(dists)
print(len(dists))
print(len(points))
print(len(weights))
spline_x = UnivariateSpline(dists, points[:, 0], k=1)#, w=weights)
spline_y = UnivariateSpline(dists, points[:, 1], k=1)#, w=weights)

d = np.linspace(dists[0], dists[-1], 10)
print("d", d)
# plt.plot(cs(dists)[:, 0], cs(dists)[:, 1], "o--", label="interp", linewidth=3)

new_x = spline_x(d)
new_y = spline_y(d)

eqpoints = np.vstack((new_x, new_y)).T
plt.plot(eqpoints[:, 0], eqpoints[:, 1], "o--", label="1d")


res = cs(d)
print("cs(d)", res)
# plt.plot(res[:, 0], res[:, 1], "--o", label="d", )

for i in range(len(res)):
    l1 = getLength(d[i:i+2])
    l2 = getLength(res[i:i+2])
    l3 = getLength(eqpoints[i:i+2])
    print(l1, l2, l3)

plt.axis('equal')
plt.legend()
plt.show()
