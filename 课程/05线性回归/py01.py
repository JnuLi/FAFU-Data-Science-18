from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing

x = np.array([ 338, 333, 328, 207, 226, 25, 179, 60, 208, 606])
y = np.array([ 640, 633, 619, 393, 428, 27, 193, 66, 226, 1591])

x = preprocessing.scale(x)
y = preprocessing.scale(y)
fig = plt.figure()
ax = fig.gca(projection='3d')

X = np.arange(0.5, 1.5, 0.01)
Y = np.arange(-0.5, 0.5, 0.01)
X, Y = np.meshgrid(X, Y)
Z = np.array([np.sum(np.fabs((X[i][j]*x + Y[i][j] - y))) for i in range(100) for j in range(100)]).reshape(100, 100)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')

plt.show()
