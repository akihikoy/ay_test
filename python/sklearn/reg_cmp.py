#!/usr/bin/python
#\file    reg_cmp.py
#\brief   Comparing regression models.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Feb.13, 2020

#Modules for GPR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

#Modules for linear regression
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression


#For data generation and plot
import random, math
import numpy as np
import matplotlib.pyplot as plot
from mpl_toolkits.mplot3d import Axes3D

def f(x):
  return x[0]+math.sin(3.0*x[1])

#Uniform noise in [-1,1]
rand= lambda: 2.0*(-0.5+random.random())

data_x= [[2.0*rand(), 2.0*rand()] for i in range(200)]
data_y= [[f(x)+0.1*rand()] for x in data_x]


#Training GPR
#kernel= C(1.0, (1e-3, 1e3)) * RBF(1.0, (0.1, 10.0))
#kernel= C(1.0, (1.0, 1.0)) * RBF(1.0, (0.1, 10.0))
#kernel= C(1.0, (1e-3, 1e3)) * RBF(3.0, (3.0, 3.0))
#kernel= RBF(1.0, (0.1, 10.0))
kernel= RBF(3.0, (3.0, 3.0))
gpr= GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
gpr.fit(data_x, data_y)

#Training linear regression
linreg= LinearRegression().fit(data_x, data_y)


#Evaluating models
def EvalDataError(model, data_x, data_y):
  return np.sqrt(np.mean((model.predict(data_x)-data_y)**2))

print 'gpr-RMSE=', EvalDataError(gpr, data_x, data_y)
print 'linreg-RMSE=', EvalDataError(linreg, data_x, data_y)


#Plot setup
XY= np.mgrid[-2:2:0.1,-2:2:0.1]
fig= plot.figure()
plot3d= Axes3D(fig)

#Plot GPR
Z= np.vectorize(lambda x,y: gpr.predict([[x,y]]))(XY[0],XY[1])
plot3d.plot_wireframe(XY[0],XY[1],Z,color=[0,1,0])

#Plot linear regression
Z= np.vectorize(lambda x,y: linreg.predict([[x,y]]))(XY[0],XY[1])
plot3d.plot_wireframe(XY[0],XY[1],Z,color=[0,0,1])

#Plot data points
plot3d.scatter(np.array(data_x).T[0],np.array(data_x).T[1],data_y,marker='*',color=[1,0,0])

plot.show()


