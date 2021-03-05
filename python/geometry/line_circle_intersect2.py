#!/usr/bin/python
#\file    line_circle_intersect2.py
#\brief   Get intersections of a line segment and a circle.
#         Application to get an intersections between 3d line and a cylinder.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Mar.06, 2021
from line_circle_intersect import *

if __name__=='__main__':
  def write_polygon(fp,polygon):
    if len(polygon)>0:
      for pt in polygon+[polygon[0]]:
        fp.write('%s\n'%' '.join(map(str,pt)))
    fp.write('\n')

  p1= np.random.uniform([-20,-20,-20],[20,20,20])
  p2= np.random.uniform([-20,-20,-20],[20,20,20])
  pc= np.random.uniform([-5,-5,0],[5,5,0])
  rad= np.random.uniform(5,20)
  t_intersects= LineCircleIntersections(p1, p2, pc, rad)
  p_intersects= [(1.0-t)*p1+t*p2 for t in t_intersects]
  print 't_intersects:',t_intersects

  poly_circle= [pc+rad*np.array([np.cos(th),np.sin(th),0]) for th in np.linspace(0,np.pi*2,100)]

  with open('/tmp/lines.dat','w') as fp:
    write_polygon(fp, [p1,p2])
    write_polygon(fp, poly_circle)
    for pI in p_intersects:
      write_polygon(fp, [pI])

  print '#Plot by:'
  print '''qplot -x -3d -s 'set size ratio -1' /tmp/lines.dat u 1:2:3:'(column(-1)+1)' w lp lc var pt 4'''
