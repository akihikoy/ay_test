#!/usr/bin/python
#\file    basic01.py
#\brief   Basic computation of Chainer.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.04, 2015

import numpy as np
from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers
import chainer.functions as F

def Main1():
  print 'Basic'
  x_data= np.array([5], dtype=np.float32)
  x= Variable(x_data)
  y= x**2 - 2 * x + 1
  print '--'
  print x.data
  print y.data
  print '--'
  print x.grad
  print y.grad
  print '--'
  y.backward()
  print x.grad
  print y.grad
  print '------------'

def Main2():
  print 'Test of if-else syntax (DOESN\'T WORK)'
  x_data= np.array([5], dtype=np.float32)
  x= Variable(x_data)
  #y= x**2 - 2 * np.sin(x) + 1  #ERROR: AttributeError: sin
  y= x**2 - 2.0 if x>0 else 0.0  #ERROR: NotImplementedError
  print '--'
  print x.data
  print y.data
  print '--'
  print x.grad
  print y.grad
  print '--'
  y.backward()
  print x.grad
  print y.grad
  print '------------'

def Main3():
  print 'Test of split and concat'
  x_data= np.array([0,1,2,3,4,5,6], dtype=np.float32)
  x= Variable(x_data)
  x1,x2= F.split_axis(x, [2], 0)
  y= F.concat((x1,x2),0)
  print '--'
  print x.data
  print x1.data
  print x2.data
  print '--'
  print y.data
  #print x.grad
  #y.backward()
  #print x.grad
  print '------------'

def Main4():
  print 'Test of multiple independent functions sharing the input'
  x_data= np.array([5], dtype=np.float32)
  x= Variable(x_data)
  y1= 2*x + 1
  y2= -1.5*x
  z1= 0.5*y1
  z2= 0.8*y2
  print '--'
  print x.data
  print y1.data
  print y2.data
  print z1.data
  print z2.data
  print '--'
  print x.grad
  print y1.grad
  print y2.grad
  print z1.grad
  print z2.grad
  print '--'
  z1.backward()
  y1.backward()
  print x.grad
  print y1.grad
  print y2.grad
  print z1.grad
  print z2.grad
  print '--'
  z2.backward()
  y2.backward()
  print x.grad
  print y1.grad
  print y2.grad
  print z1.grad
  print z2.grad
  print '------------'

def Main5():
  print 'Test of multiple independent functions sharing the input'
  x_data= np.array([[1,2,3]], dtype=np.float32).T
  W_data= np.array([[1,2,0],[0,2,1],[0,0,1]], dtype=np.float32)
  b= np.array([3,2,1], dtype=np.float32)
  x= Variable(x_data)
  W= Variable(W_data)
  #y= x*W.T + b
  #y= F.matmul(x,W,transa=True,transb=True)# + b
  y= F.matmul(W,x)# + b
  y.grad= np.array([[1,1,1]], dtype=np.float32).T
  #y.grad= np.array([[1,0,0],[0,1,0],[0,0,1]], dtype=np.float32).T
  print '--'
  print x.data
  #print W.data
  print y.data
  #print x_data.dot(W_data.T)# + b
  print W_data.dot(x_data)# + b
  print '--'
  y.backward()
  print x.grad
  #print W.grad
  print y.grad
  print '------------'


if __name__=='__main__':
  Main1()
  #Main2()
  Main3()
  Main4()
  Main5()

