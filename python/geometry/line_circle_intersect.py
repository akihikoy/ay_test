#!/usr/bin/python
#\file    line_circle_intersect.py
#\brief   Get intersections of a line segment and a circle.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Mar.06, 2021
import numpy as np

#Get intersections of a line segment (p1,p2) and a circle (pc,rad).
#  Return [t1,t2] or [t1] or [] where tN is a scolar representing an intersection:
#    p= (1.0-t)*p1+t*p2
def LineCircleIntersections(p1, p2, pc, rad):
  p,q= p2[0]-p1[0],p2[1]-p1[1]
  r,s= pc[0]-p1[0],pc[1]-p1[1]
  len12_sq= p*p+q*q
  DET= len12_sq*rad*rad - (p*s-q*r)**2
  if DET<0:  return []
  elif DET==0:
    t= (r*p+s*q)/len12_sq
    if 0<=t<=1:  return [t]
    return []
  else:
    rp_sq= r*p+s*q
    DET= np.sqrt(DET)
    t1= (rp_sq-DET)/len12_sq
    t2= (rp_sq+DET)/len12_sq
    sol= []
    if 0<=t1<=1:  sol.append(t1)
    if 0<=t2<=1:  sol.append(t2)
    return sol


if __name__=='__main__':
  def write_polygon(fp,polygon):
    if len(polygon)>0:
      for pt in polygon+[polygon[0]]:
        fp.write('%s\n'%' '.join(map(str,pt)))
    fp.write('\n')

  p1= np.random.uniform([-20,-20],[20,20])
  p2= np.random.uniform([-20,-20],[20,20])
  pc= np.random.uniform([-5,-5],[5,5])
  rad= np.random.uniform(5,20)
  t_intersects= LineCircleIntersections(p1, p2, pc, rad)
  p_intersects= [(1.0-t)*p1+t*p2 for t in t_intersects]
  print 't_intersects:',t_intersects

  poly_circle= [pc+rad*np.array([np.cos(th),np.sin(th)]) for th in np.linspace(0,np.pi*2,100)]

  with open('/tmp/lines.dat','w') as fp:
    write_polygon(fp, [p1,p2])
    write_polygon(fp, poly_circle)
    for pI in p_intersects:
      write_polygon(fp, [pI])

  print '#Plot by:'
  print '''qplot -x -s 'set size ratio -1' /tmp/lines.dat u 1:2:'(column(-1)+1)' w lp lc var pt 4'''

