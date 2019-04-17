#!/usr/bin/python
#\file    fk_test2.py
#\brief   Test learned forward kinematics (decomposed model).
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.13, 2015

import joint_chain1 as sim
import time,sys
import math,random
from fk_test import TFKTester, AddVizCube

def Main():
  import argparse
  import re
  import numpy as np
  import numpy.linalg as la
  from chainer import cuda, Variable
  import chainer.functions  as F
  import six.moves.cPickle as pickle
  from fk_learn2 import ForwardModel, ModelCodesWithXAll

  parser = argparse.ArgumentParser(description='Chainer example: regression')
  parser.add_argument('--gpu', '-g', default=-1, type=int,
                      help='GPU ID (negative value indicates CPU)')
  parser.add_argument('--dof', '-D', default='3', type=str,
                      help='DoF code')
  parser.add_argument('--mdof', '-MD', default='', type=str,
                      help='DoF code of model file. Blank uses the same one as --dof.')
  args = parser.parse_args()

  #dof = 3
  dofc= args.dof
  mdofc= args.mdof if args.mdof!='' else dofc
  dof= int(re.search('^[0-9]+',dofc).group())

  # Load model from file
  def load_model():
    global model1, model2
    model = pickle.load(open('result/fk2nn%s.dat'%mdofc, 'rb'))
    print 'Loaded model from:','result/fk2nn%s.dat'%mdofc
    model1, model2= model['model1'], model['model2']
    if args.gpu >= 0:
      cuda.init(args.gpu)
      model1.to_gpu()
      model2.to_gpu()
  load_model()

  Dy1= 3 if dofc not in ModelCodesWithXAll else 3*dof
  Dy2= 4 if dofc not in ModelCodesWithXAll else 4*dof

  # Setup tester
  tester= TFKTester(dof)

  # Neural net architecture
  def forward(x_data, y1_data, y2_data, train=True):
    return ForwardModel(dofc, model1, model2, x_data, y1_data, y2_data, train)

  # Predict for a single query x
  def predict(x):
    x_batch = np.array([x]).astype(np.float32)
    y1_batch = np.array([[0.0]*Dy1]).astype(np.float32)  #Dummy
    y2_batch = np.array([[0.0]*Dy2]).astype(np.float32)  #Dummy
    if args.gpu >= 0:
      x_batch = cuda.to_gpu(x_batch)
      y1_batch = cuda.to_gpu(y1_batch)
      y2_batch = cuda.to_gpu(y2_batch)
    loss1,loss2, pred1,pred2 = forward(x_batch, y1_batch, y2_batch, train=False)
    y1= cuda.to_cpu(pred1.data)[0]
    y2= cuda.to_cpu(pred2.data)[0]
    y= np.concatenate((y1,y2))
    #y[-4:]= [0,0,0,1]  #Modify the orientation
    y[-4:]/= la.norm(y[-4:])  #Normalize the quaternion
    return y

  for i in range(10):
    load_model()
    print predict(np.array([0.0]*dof).astype(np.float32))
    tester.Test(f_fwdkin=predict, n_samples=100)

  tester.Cleanup()


def PlotGraphs():
  print 'Plotting graphs..'
  import os
  commands=[
    '''qplot -x2 aaa 'sin(x)' w l &''',
    '''''',
    '''''',
    ]
  for cmd in commands:
    if cmd!='':
      cmd= ' '.join(cmd.splitlines())
      print '###',cmd
      os.system(cmd)

  print '##########################'
  print '###Press enter to close###'
  print '##########################'
  raw_input()
  os.system('qplot -x2kill aaa')

if __name__=='__main__':
  import sys
  if len(sys.argv)>1 and sys.argv[1] in ('p','plot','Plot','PLOT'):
    PlotGraphs()
    sys.exit(0)
  Main()
