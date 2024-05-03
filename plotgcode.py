import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys, os.path
np.set_printoptions(suppress=True)

# gen
data = np.zeros((4,), dtype=float)

if(len(sys.argv) > 1):
    if(os.path.isfile(sys.argv[1])):
        name = sys.argv[1]
else:
    name = "left-wing-segment-2.ngc"

file = open(name, "r")

for line in file:
    if line[0:2] == "G1":
        d = line.split()
        if(len(d) == 5):
            arr = np.empty((4,), dtype=float)
            # print(d)
            for i in range(4):
                arr[i] = d[i+1][1:]
            # print(arr)
            data = np.vstack([data, arr])
    else:
        if(line[0] != '(' and line[0] != '%'):
            print("not recognized: ", line[:-1])

# print("data", data)

# plot
def plotStatic():
    i = 0
    plt.plot(data[i:, 0], data[i:, 1])
    plt.plot(data[i:, 2], data[i:, 3], "--")
    plt.axis('equal')

    plt.show()

def plotDynamic():
    def init():
        # plt.axis('equal')
        plt.legend(['Line 1', 'Line 2'])
        return []

    dists = np.sqrt(np.square(((data[1:, 0]-data[:-1, 0])-(data[1:, 1]-data[:-1, 1]))))
    lmax = max(np.max(data[:, 0])*1.1, np.max(data[:, 1])*1.1)

    def update(frame):
        if frame < len(data[:])-1:
            plt.clf()
            plt.xlim(-0.1*lmax, lmax)
            plt.ylim(-0.1*lmax, lmax)
            plt.plot(data[0:frame+1, 0], data[0:frame+1, 1])
            plt.plot(data[0:frame+1, 2], data[0:frame+1, 3], "--")
        return []        


    ani = FuncAnimation(plt.gcf(), update, frames=len(data[:]), init_func=init, interval=100, blit=True)

    plt.show()

    plt.close()
    
if __name__ == "__main__":
    plotStatic()
