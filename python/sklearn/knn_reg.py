#!/usr/bin/python3
#\file    knn_reg.py
#\brief   Nearest Neighbor for Regression
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Mar.12, 2020

import numpy as np
import matplotlib.pyplot as plot
from mpl_toolkits.mplot3d import Axes3D

from sklearn import neighbors

def f(x):
  return x[0]+np.sin(3.0*x[1])

rand= lambda: np.random.uniform(-1.0,1.0)

data_x= [[2.0*rand(), 2.0*rand()] for i in range(200)]
data_y= [[f(x)+0.1*rand()] for x in data_x]


knn= neighbors.KNeighborsRegressor(n_neighbors=5, weights='distance')  #weights in ['uniform', 'distance']
reg= knn.fit(data_x, data_y)

print('score:',reg.score(data_x, data_y))

#Now we can estimate y for a given x by
#  y_pred= reg.predict(x)
print(reg.predict([[1.0,1.0]]))


#Plot reg.predict(x)
XY= np.mgrid[-2:2:0.1,-2:2:0.1]
Z= np.vectorize(lambda x,y: reg.predict([[x,y]]))(XY[0],XY[1])
fig= plot.figure()
plot3d= Axes3D(fig)
plot3d.plot_wireframe(XY[0],XY[1],Z)

#Plot data points
plot3d.scatter(np.array(data_x).T[0],np.array(data_x).T[1],data_y,marker='*',color='red')

plot.show()
