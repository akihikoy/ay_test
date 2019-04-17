#!/usr/bin/python
#\file    intplcalc1.py
#\brief   Test of intplcalc.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jul.13, 2016

def Main():
  import os
  os.system('''intplcalc -f a data/vsfL4.dat 1,2 -f b data/vsfR4.dat 1,2 'a[1]+b[1]' 'a[1]*b[1]' > /tmp/ic1.dat ''')

def PlotGraphs():
  print 'Plotting graphs..'
  import os
  commands=[
    '''qplot -x2 aaa
        data/vsfL4.dat u 1:2 w p
        data/vsfR4.dat u 1:2 w p
        /tmp/ic1.dat u 1:2 w l
        /tmp/ic1.dat u 1:3 w l
        &''',
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
