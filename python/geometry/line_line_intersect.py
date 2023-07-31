#!/usr/bin/python
#\file    line_line_intersect.py
#\brief   Get intersection of two line segments.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.20, 2020
import math
import numpy as np

#Return an intersection between line seg (p1,p2) and line seg (pA,pB).
#Return None if there is no intersection.
#Based on: https://www.cs.hmc.edu/ACM/lectures/intersections.html
def LineLineIntersection(p1, p2, pA, pB, tol=1e-8, return_rs=False):
  x1, y1 = p1;   x2, y2 = p2
  dx1 = x2 - x1;  dy1 = y2 - y1
  xA, yA = pA;   xB, yB = pB;
  dxA = xB - xA;  dyA = yB - yA;

  DET = (-dx1 * dyA + dy1 * dxA)
  if math.fabs(DET) < tol: return None

  DETinv = 1.0/DET
  r = DETinv * (-dyA * (xA-x1) + dxA * (yA-y1))
  s = DETinv * (-dy1 * (xA-x1) + dx1 * (yA-y1))
  if r<0.0 or s<0.0 or r>1.0 or s>1.0:  return None

  if return_rs:  return [r,s]
  xi = (x1 + r*dx1 + xA + s*dxA)/2.0
  yi = (y1 + r*dy1 + yA + s*dyA)/2.0
  return [xi,yi]

#Return an intersection between inf line (p1+r*dp1) and line seg (pA,pB).
#Return None if there is no intersection.
def InfLineLineIntersection(p1, dp1, pA, pB, tol=1e-8, return_rs=False):
  x1, y1 = p1
  dx1, dy1 = dp1
  xA, yA = pA;   xB, yB = pB;
  dxA = xB - xA;  dyA = yB - yA;

  DET = (-dx1 * dyA + dy1 * dxA)
  if math.fabs(DET) < tol: return None

  DETinv = 1.0/DET
  r = DETinv * (-dyA * (xA-x1) + dxA * (yA-y1))
  s = DETinv * (-dy1 * (xA-x1) + dx1 * (yA-y1))
  if s<0.0 or s>1.0:  return None

  if return_rs:  return [r,s]
  xi = (x1 + r*dx1 + xA + s*dxA)/2.0
  yi = (y1 + r*dy1 + yA + s*dyA)/2.0
  return [xi,yi]

def Main():
  def write_polygon(fp,polygon):
    if len(polygon)>0:
      for pt in polygon+[polygon[0]]:
        fp.write('%s\n'%' '.join(map(str,pt)))
    fp.write('\n')

  p1= np.random.uniform([-10,-10],[10,10])
  p2= np.random.uniform([-10,-10],[10,10])
  dp1= (p2-p1); dp1/= np.linalg.norm(dp1)
  pA= np.random.uniform([-10,-10],[10,10])
  pB= np.random.uniform([-10,-10],[10,10])
  print '#points:',p1, p2, pA, pB
  pI= LineLineIntersection(p1, p2, pA, pB)
  pI2= InfLineLineIntersection(p1, dp1, pA, pB)
  print 'line-line-intersection:',pI
  print 'infline-line-intersection:',pI2

  with open('/tmp/lines.dat','w') as fp:
    write_polygon(fp,[p1, p2])
    write_polygon(fp,[pA, pB])
    if pI is not None:  write_polygon(fp,[pI])
    if pI2 is not None:  write_polygon(fp,[pI2])

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
