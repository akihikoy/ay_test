#!/usr/bin/python
#\file    numpy.interp.py
#\brief   Test of numpy.interp;
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jun.16, 2023
import time
import numpy as np

data_file= 'data/vsfL11.dat'  #584 points
'''
@20230616@ubuntu
Done linear interpolation/test in 9.10758972168e-05 sec
'''

#data_file= 'data/vsfL4.dat'  #4045 points
'''
@20230616@ubuntu
Done linear interpolation/test in 0.000363111495972 sec
'''

#Float version of range
def FRange1(x1,x2,num_div):
  return [x1+(x2-x1)*x/float(num_div) for x in range(num_div+1)]

def Main():
  fp= open(data_file)
  X= []
  Y= []
  while True:
    line= fp.readline()
    if not line: break
    values= line.split()
    X.append(float(values[0]))
    Y.append(float(values[1]))
  fp.close()
  xmin= X[0]
  xmax= X[-1]

  X_test= [x for x in FRange1(xmin,xmax,1000)]

  t_start= time.time()
  Y_test= np.interp(X_test, X, Y)
  print 'Done linear interpolation/test in {t} sec'.format(t=time.time()-t_start)


  def save(XX, YY, file_name):
    fp= open(file_name,'w')
    for x,y in zip (XX,YY):
      fp.write('{x} {y}\n'.format(x=x,y=y))
    fp.close()

  save(X_test,Y_test,'/tmp/npinterp.dat')

def PlotGraphs():
  print 'Plotting graphs..'
  import os
  commands=[
    '''qplot -x2 aaa
      {data_file} w p
      /tmp/npinterp.dat w l &
      '''.format(data_file=data_file),
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
