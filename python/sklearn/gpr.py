#!/usr/bin/python
#NOTE: Remove and Upgrade scikit-learn (v.1.8 is necessary)
#  sudo apt-get remove python-sklearn python-sklearn-lib
#  sudo pip install -U scikit-learn

import random, math
import numpy as np
import matplotlib.pyplot as plot
from mpl_toolkits.mplot3d import Axes3D
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

def f(x):
  return x[0]*math.sin(3.0*x[1])

#Uniform noise in [-1,1]
rand= lambda: 2.0*(-0.5+random.random())

data_x= [[2.0*rand(), 2.0*rand()] for i in range(200)]
data_y= [[f(x)+0.1*rand()] for x in data_x]

#kernel= C(1.0, (1e-3, 1e3)) * RBF(1.0, (0.1, 10.0))
#kernel= C(1.0, (1.0, 1.0)) * RBF(1.0, (0.1, 10.0))
#kernel= C(1.0, (1e-3, 1e3)) * RBF(3.0, (3.0, 3.0))
#kernel= RBF(1.0, (0.1, 10.0))
kernel= RBF(3.0, (3.0, 3.0))
gp= GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
gp.fit(data_x, data_y)
#Now we can estimate y for a given x by
#  y_pred= gp.predict(x)
print gp.predict([[1.0,1.0]])

#Plot gp.predict(x)
XY= np.mgrid[-2:2:0.1,-2:2:0.1]
Z= np.vectorize(lambda x,y: gp.predict([[x,y]]))(XY[0],XY[1])
fig= plot.figure()
plot3d= Axes3D(fig)
plot3d.plot_wireframe(XY[0],XY[1],Z)

#Plot data points
plot3d.scatter(np.array(data_x).T[0],np.array(data_x).T[1],data_y,marker='*',color='red')

plot.show()

