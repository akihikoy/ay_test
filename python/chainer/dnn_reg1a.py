#!/usr/bin/python
#\file    dnn_reg1a.py
#\brief   Example of TNNRegression.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.24, 2015
from dnn_reg1 import TNNRegression, DumpPlot, SaveYAML, LoadYAML

def LoadData():
  src_file= 'data/ode_f1_3_smp.dat'; dim= [2,5,5]
  data_x= []
  data_y= []
  fp= file(src_file)
  while True:
    line= fp.readline()
    if not line: break
    data= line.split()
    data_x.append(map(float,data[sum(dim[0:1]):sum(dim[0:2])]))
    data_y.append(map(float,data[sum(dim[0:2]):sum(dim[0:3])]))
  return data_x, data_y


def Main():
  #def PrintEq(s):  print '%s= %r' % (s, eval(s))
  data_x, data_y = LoadData()

  options= {}
  options['n_units']= [5,200,200,200,5]

  load_model,train_model= False,True
  #load_model,train_model= True,False
  model= TNNRegression()
  model.Load(data={'options':options})
  if load_model:
    model.Load(LoadYAML('/tmp/dnn/nn_model.yaml'), '/tmp/dnn/')
  model.Init()
  if train_model:
    for x,y,n in zip(data_x,data_y,range(len(data_x))):
      print '========',n,'========'
      model.Update(x,y,not_learn=((n+1)%10!=0))
    #model.Update()
    #model.UpdateBatch(data_x,data_y)

  print 'model.NSamples=',model.NSamples
  print 'model.Dx=',model.Dx
  print 'model.Dy=',model.Dy

  if not load_model:
    SaveYAML(model.Save('/tmp/dnn/'), '/tmp/dnn/nn_model.yaml')

  f_reduce=lambda xa:[xa[0],xa[3]]
  f_repair=lambda xa,mi,ma,me:[xa[0],me[1],me[2],xa[1],me[4]]
  DumpPlot(model, f_reduce=f_reduce, f_repair=f_repair, x_var=0.0, file_prefix='/tmp/dnn/f')


def PlotGraphs():
  print 'Plotting graphs..'
  import os
  commands=[
    '''qplot -x2 aaa -3d
          -s 'set xlabel "rcv_x";set ylabel "pour_x";set title "dpour_x";set ticslevel 0;'
          -cs 'u 1:2:11'
          /tmp/dnn/f_est.dat w l lw 2 t '"prediction"'
          /tmp/dnn/f_smp.dat pt 6 ps 2 t '"sample"'  &''',
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
