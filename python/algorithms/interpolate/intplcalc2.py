#!/usr/bin/python3
#\file    intplcalc1.py
#\brief   Test of intplcalc.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jul.13, 2016

#Float version of range
def FRange1(x1,x2,num_div):
  return [x1+(x2-x1)*x/float(num_div) for x in range(num_div+1)]

def Main():
  import os

  fp= open('/tmp/test1.dat','w')
  for x in FRange1(0,10.0,1000):
    fp.write('{x} {y1}\n'.format(x=x,y1=x))
  fp.close()

  fp= open('/tmp/test2.dat','w')
  for x in FRange1(0,10.0,200):
    fp.write('{x} {y1} {y2}\n'.format(x=x,y1=0.5*x,y2=2.0))
  fp.close()

  os.system('''intplcalc -f a /tmp/test1.dat 1,2 -f b /tmp/test2.dat 1,2,3 'a[1]+b[1]' 'a[1]*b[1]*0.4' 'a[1]+b[2]' 'a[1]*b[2]' > /tmp/ic2.dat ''')

def PlotGraphs():
  print('Plotting graphs..')
  import os
  commands=[
    '''qplot -x2 aaa
        /tmp/test1.dat u 1:2 w p t '"y1=x"'
        /tmp/test2.dat u 1:2 w p t '"y2=0.5*x"'
        /tmp/test2.dat u 1:3 w p t '"y3=2.0"'
        /tmp/ic2.dat u 1:2 w l lw 2 t '"y=y1+y2"'
        /tmp/ic2.dat u 1:3 w l lw 2 t '"y=y1*y2*0.4"'
        /tmp/ic2.dat u 1:4 w l lw 2 t '"y=y1+y3"'
        /tmp/ic2.dat u 1:5 w l lw 2 t '"y=y1*y3"'
        &''',
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
