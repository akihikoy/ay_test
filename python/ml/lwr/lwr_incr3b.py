#!/usr/bin/python3
#\file    lwr_incr3b.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jun.16, 2015

from lwr_incr2 import *
#from lwr_incr3 import *

#Float version of range
def FRange1(x1,x2,num_div):
  return [x1+(x2-x1)*x/float(num_div) for x in range(num_div+1)]

def Main():
  #def PrintEq(s):  print '%s= %r' % (s, eval(s))

  model= TLWR(kernel='maxg')
  #model.Init(c_min=0.6, f_reg=0.00001)
  #model.Init(c_min=0.3, f_reg=0.001)
  #model.Init(c_min=0.1, f_reg=0.001)
  #model.Init(c_min=0.01, f_reg=0.01)
  #model.Init(c_min=0.01, c_max=0.1, f_reg=0.0001)
  model.Init(c_min=0.01, f_reg=0.0001)
  #model.Init(c_min=0.03, f_reg=0.0001)
  #model.Init(c_min=0.002, f_reg=0.001)
  #model.Init(c_min=0.0001, f_reg=0.0000001)
  #model.Init(c_min=1.0e-6, f_reg=0.1)
  src_file= 'data/ode_f1_3_smp.dat'; dim= [2,5,5]
  #assess= lambda y: 5.0*y[0]+y[1]

  fp= open(src_file)
  while True:
    line= fp.readline()
    if not line: break
    data= line.split()
    model.Update(list(map(float,data[sum(dim[0:1]):sum(dim[0:2])])),
                 list(map(float,data[sum(dim[0:2]):sum(dim[0:3])])))
  #model.C= [0.1]*len(model.C)
  #model.C= [0.03]*len(model.C)
  #model.C= [0.01]*len(model.C)
  #model.C= [0.001]*len(model.C)
  #model.C= [0.0001]*len(model.C)
  #model.C= model.AutoWidth(model.CMin)
  #model.C= [c+0.01 for c in model.C]
  importance= {}
  model.Importance= importance
  #for i in range(len(model.DataY)):
    #if assess(model.DataY[i])>0.5:
      ##model.C[i]= 0.1
      #importance[i]= 5.0*assess(model.DataY[i])
      #print importance[i]

  mi= [min([x[d] for x in model.DataX]) for d in range(len(model.DataX[0]))]
  ma= [max([x[d] for x in model.DataX]) for d in range(len(model.DataX[0]))]
  me= [Median([x[d] for x in model.DataX]) for d in range(len(model.DataX[0]))]
  #mi[0]= -0.6
  #ma[2]= 1.0

  #"""
  f_reduce=lambda xa:[xa[0],xa[3]]
  f_repair=lambda xa,mi,ma,me:[xa[0],me[1],me[2],xa[1],me[4]]
  model.DumpPlot(bounds=[mi,ma], x_var=[0.,0.,0.,0.,0.], f_reduce=f_reduce, f_repair=f_repair, file_prefix='/tmp/lwr/f1_3')
  #fp= open('/tmp/lwr/f3_ideals.dat','w')
  #for xa1,x2 in zip(model.DataX, model.DataY):
    #if assess(x2)>0.5:
      #fp.write('%s\n' % ToStr(f_reduce(xa1),xa1,[assess(x2)]))
  #fp.close()
  #"""

def PlotGraphs():
  print('Plotting graphs..')
  import os
  commands=[
    '''qplot -x2 aaa -3d
          -s 'set xlabel "rcv_x";set ylabel "pour_x";set title "dpour_x";set ticslevel 0;'
          -cs 'u 1:2:11' /tmp/lwr/f1_3_est.dat w l /tmp/lwr/f1_3_smp.dat -cs '' &''',
    '''''',
    '''''',
    ]
  for cmd in commands:
    if cmd!='':
      cmd= ' '.join(cmd.splitlines())
      print('###',cmd)
      os.system(cmd)

  print('##########################')
  print('###Press enter to close###')
  print('##########################')
  input()
  os.system('qplot -x2kill aaa')

if __name__=='__main__':
  import sys
  if len(sys.argv)>1 and sys.argv[1] in ('p','plot','Plot','PLOT'):
    PlotGraphs()
    sys.exit(0)
  Main()
