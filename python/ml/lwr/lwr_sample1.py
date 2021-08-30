#!/usr/bin/python

'''
ref.
http://vilkeliskis.com/blog/2013/09/08/machine_learning_part_2_locally_weighted_linear_regression.html
'''

import numpy as np

def gaussian_kernel(x, x0, c, a=1.0):
  """
  Gaussian kernel.

  :Parameters:
    - `x`: nearby datapoint we are looking at.
    - `x0`: data point we are trying to estimate.
    - `c`, `a`: kernel parameters.
  """
  # Euclidian distance
  diff = x - x0
  dot_product = diff * diff.T
  return a * np.exp(dot_product / (-2.0 * c**2))


def get_weights(training_inputs, datapoint, c=1.0):
  """
  Function that calculates weight matrix for a given data point and training
  data.

  :Parameters:
    - `training_inputs`: training data set the weights should be assigned to.
    - `datapoint`: data point we are trying to predict.
    - `c`: kernel function parameter

  :Returns:
    NxN weight matrix, there N is the size of the `training_inputs`.
  """
  x = np.mat(training_inputs)
  n_rows = x.shape[0]
  # Create diagonal weight matrix from identity matrix
  weights = np.mat(np.eye(n_rows))
  for i in xrange(n_rows):
    weights[i, i] = gaussian_kernel(datapoint, x[i], c)

  return weights


def lwr_predict(training_inputs, training_outputs, datapoint, c=1.0, f_reg=0.01):
  """
  Predict a data point by fitting local regression.

  :Parameters:
    - `training_inputs`: training input data.
    - `training_outputs`: training outputs.
    - `datapoint`: data point we want to predict.
    - `c`: kernel parameter.

  :Returns:
    Estimated value at `datapoint`.
  """
  weights = get_weights(training_inputs, datapoint, c=c)

  x = np.mat(training_inputs)
  y = np.mat(training_outputs)

  xt = x.T * (weights * x)
  betas = (xt+f_reg*np.eye(xt.shape[0])).I * (x.T * (weights * y))

  return datapoint * betas



import random

#Float version of range
def FRange1(x1,x2,num_div):
  return [x1+(x2-x1)*x/float(num_div) for x in range(num_div+1)]

def Rand(xmin=-0.5,xmax=0.5):
  return random.random()*(xmax-xmin)+xmin


if __name__=='__main__':
  #def PrintEq(s):  print '%s= %r' % (s, eval(s))
  import math
  true_func= lambda x: 1.2+math.sin(x)
  data_x= [[x+1.0*Rand(),1.0] for x in FRange1(-3.0,5.0,10)]  # ,1.0 is to learn const
  data_y= [[true_func(x[0])+0.3*Rand()] for x in data_x]

  fp1= file('/tmp/smpl.dat','w')
  for x,y in zip(data_x,data_y):
    fp1.write('%f %f\n' % (x[0],y[0]))
  fp1.close()

  fp1= file('/tmp/true.dat','w')
  fp2= file('/tmp/est.dat','w')
  for x in FRange1(-7.0,10.0,200):
    y= lwr_predict(data_x, data_y, np.mat([x,1.0]), c=0.5)  # ,1.0 is to learn const
    fp1.write('%f %f\n' % (x,true_func(x)))
    fp2.write('%f %f\n' % (x,y))
  fp1.close()
  fp2.close()

  print 'Plot by:'
  print 'qplot -x /tmp/est.dat w l /tmp/true.dat w l /tmp/smpl.dat w p'

