#!/usr/bin/python
#\file    line_line_intersect2.py
#\brief   Check if two line segments have an intersection.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jul.30, 2023

#Check if two line segments (p1-p2, pA-pB) have an intersection.
#ref. https://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/
def DoLineLineIntersect(p1, p2, pA, pB):
  def ccw(a, b, c):
    return (c[1]-a[1])*(b[0]-a[0]) > (b[1]-a[1])*(c[0]-a[0])
  return ccw(p1,pA,pB)!=ccw(p2,pA,pB) and ccw(p1,p2,pA)!=ccw(p1,p2,pB)


def Main():
  from line_line_intersect import LineLineIntersection
  import numpy as np
  def write_polygon(fp,polygon):
    if len(polygon)>0:
      for pt in polygon+[polygon[0]]:
        fp.write('%s\n'%' '.join(map(str,pt)))
    fp.write('\n')

  p1= np.random.uniform([-10,-10],[10,10])
  p2= np.random.uniform([-10,-10],[10,10])
  pA= np.random.uniform([-10,-10],[10,10])
  pB= np.random.uniform([-10,-10],[10,10])
  print '#points:',p1, p2, pA, pB
  pI= LineLineIntersection(p1, p2, pA, pB)
  print '##########################'
  print '###intersection:',pI, DoLineLineIntersect(p1, p2, pA, pB)
  print '##########################'

  with open('/tmp/lines.dat','w') as fp:
    write_polygon(fp,[p1, p2])
    write_polygon(fp,[pA, pB])
    if pI is not None:  write_polygon(fp,[pI])

def PlotGraphs():
  print 'Plotting graphs..'
  import os
  commands=[
    '''qplot -x2 aaa
        /tmp/lines.dat u 1:2:'(column(-1)+1)' w lp lc var pt 4
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

  PlotGraphs()
  sys.exit(0)
