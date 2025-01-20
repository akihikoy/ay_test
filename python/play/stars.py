#!/usr/bin/python3
#\file    stars.py
#\brief   Draw beautiful stars.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.29, 2015
from math import sin, cos, pi

def Shape1(r=1.0, N=5, t=1, x=[0.0,0.0,0.0]):
  print(x[0],x[1])
  for i in range(N):
    x[0]+= r*cos(x[2])
    x[1]+= r*sin(x[2])
    if   t==1:  x[2]-= 2.0*pi/N
    elif t==2:  x[2]-= 2.0*pi-4.0*pi/N
    elif t==3:  x[2]-= pi-pi/N
    print(x[0],x[1])
  return x

def Shape2(r1=1.0, N1=5, t1=3, N2=3, x=[0.0,0.0,0.0]):
  for i in range(N2):
    x= Shape1(r=r1,N=N1,t=t1,x=x)
    x[2]-= 2.0*pi/N2
  return x

def Shape3(r1=1.0, N1=5, t1=3, N2=3, r3=1.0, N3=6, x=[0.0,0.0,0.0]):
  for i in range(N3):
    x= Shape2(r1=r1,N1=N1,t1=t1,N2=N2,x=x)
    x[0]+= r3*cos(x[2])
    x[1]+= r3*sin(x[2])
    x[2]-= 2.0*pi/N3
    print('')
  return x

def Main():
  import sys
  sys.stdout= open('/tmp/stars1.dat', 'w')  #Write to a file w print
  x= [0.0, 0.0, pi/2.0]  #x,y,theta
  #Shape1(r=1.0,N=7,t=3,x=x)
  #Shape2(r1=1.0,N1=7,t1=3,N2=5,x=x)
  Shape3(r1=0.25, N1=5, t1=3, N2=6, r3=0.4, N3=10, x=x)

  sys.stdout= open('/tmp/stars2.dat', 'w')  #Write to a file w print
  x= [0.0, 0.0, pi/2.0]  #x,y,theta
  Shape3(r1=0.3, N1=5, t1=1, N2=6, r3=0.4, N3=4, x=x)


def PlotGraphs():
  print('Plotting graphs..')
  import os
  commands=[
    '''qplot -x2 aaa -s 'set size ratio -1;unset tics'
        '/tmp/stars1.dat' w l lt 3 t '""'
        &''',  # -o fig/stars1.png
    '''qplot -x2 aaa -s 'set size ratio -1;unset tics'
        '/tmp/stars2.dat' w l lt 1 t '""'
        &''',
    '''''',
    ]
  for cmd in commands:
    if cmd!='':
      cmd= ' '.join(cmd.splitlines())
      print('###',cmd)
      os.system(cmd)

  input()
  os.system('qplot -x2kill aaa')

if __name__=='__main__':
  import sys
  if len(sys.argv)>1 and sys.argv[1] in ('p','plot','Plot','PLOT'):
    PlotGraphs()
    sys.exit(0)
  Main()
