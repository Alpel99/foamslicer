import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, make_interp_spline
np.set_printoptions(suppress=True)


points = np.genfromtxt("files/points.txt")
dists = np.genfromtxt("files/dists.txt")
allpoints = np.genfromtxt("files/allpoints.txt")
# print(allpoints)
print(dists)
print(points[0], points[-1])
plt.plot(points[:, 0], points[:, 1], label="orig", linewidth=4)

# cs = CubicSpline(dists, points)
cs = make_interp_spline(dists, points, k=1)


d = np.linspace(dists[0], dists[-1], 10)
print("d", d)
plt.plot(cs(dists)[:, 0], cs(dists)[:, 1], "--", label="interp", linewidth=3)
print("cs(d)", cs(d))
plt.plot(cs(d)[:, 0], cs(d)[:, 1], "--", label="d", )

plt.legend()
plt.show()
