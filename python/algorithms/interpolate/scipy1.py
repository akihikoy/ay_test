#!/usr/bin/python3
#\file    scipy1.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jul.13, 2016
import time
from scipy.interpolate import interp1d

data_file= 'data/vsfL11.dat'  #584 points
'''
@20230616@ubuntu
Done linear interpolation in 0.000228881835938 sec
Done cubic interpolation in 0.000833034515381 sec
Done linear test in 0.0281448364258 sec
Done cubic test in 0.0956690311432 sec
'''

#data_file= 'data/vsfL4.dat'  #4045 points
'''
@20230616@ubuntu
Done linear interpolation in 0.00118398666382 sec
Done cubic interpolation in 0.00266194343567 sec
Done linear test in 0.0470418930054 sec
Done cubic test in 0.0692341327667 sec
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

  t_start= time.time()
  f1= interp1d(X, Y)
  print('Done linear interpolation in {t} sec'.format(t=time.time()-t_start))

  t_start= time.time()
  f2= interp1d(X, Y, kind='cubic')
  print('Done cubic interpolation in {t} sec'.format(t=time.time()-t_start))

  def test(f,file_name):
    fp= open(file_name,'w')
    for x in FRange1(xmin,xmax,1000):
      y= f(x)
      fp.write('{x} {y}\n'.format(x=x,y=y))
    fp.close()

  t_start= time.time()
  test(f1,'/tmp/spintpl1.dat')
  print('Done linear test in {t} sec'.format(t=time.time()-t_start))

  t_start= time.time()
  test(f2,'/tmp/spintpl2.dat')
  print('Done cubic test in {t} sec'.format(t=time.time()-t_start))

def PlotGraphs():
  print('Plotting graphs..')
  import os
  commands=[
    '''qplot -x2 aaa
      {data_file} w p
      /tmp/spintpl1.dat w l
      /tmp/spintpl2.dat w l &
      '''.format(data_file=data_file),
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
