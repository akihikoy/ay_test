#!/usr/bin/python
#\file    fk_test3.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.29, 2015

from fk_test import TFKTester, AddVizCube

def Main():
  from dnn_reg1 import TNNRegression, DumpPlot, SaveYAML, LoadYAML
  import os,re
  import numpy as np
  import numpy.linalg as la
  import argparse

  #from fk_test import TFKTester

  parser = argparse.ArgumentParser(description='Chainer example: regression')
  parser.add_argument('--gpu', '-g', default=-1, type=int,
                      help='GPU ID (negative value indicates CPU)')
  parser.add_argument('--dof', '-D', default='3', type=str,
                      help='DoF code')
  parser.add_argument('--sdof', '-SD', default='', type=str,
                      help='DoF code of samples. Blank uses the same one as --dof.')
  parser.add_argument('--mdof', '-MD', default='', type=str,
                      help='DoF code of model file. Blank uses the same one as --dof.')
  args = parser.parse_args()

  #dof = 3
  dofc= args.dof
  mdofc= args.mdof if args.mdof!='' else dofc
  dof= int(re.search('^[0-9]+',dofc).group())

  file_prefix= 'result/fk3nn%s_'%mdofc
  file_names= {
    '1': '{base}1_model.yaml'.format(base=file_prefix),
    '2': '{base}2_model.yaml'.format(base=file_prefix) }

  model1= TNNRegression()
  model1.Load(LoadYAML(file_names['1']), file_prefix)
  model1.Init()
  model2= TNNRegression()
  model2.Load(LoadYAML(file_names['2']), file_prefix)
  model2.Init()

  def predict(x):
    pred1= model1.Predict(x)
    pred2= model2.Predict(x)
    y= np.concatenate((pred1.Y.ravel(),pred2.Y.ravel()))
    y[-4:]/= la.norm(y[-4:])  #Normalize the quaternion
    return y

  # Setup tester
  tester= TFKTester(dof)
  for i in range(10):
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
