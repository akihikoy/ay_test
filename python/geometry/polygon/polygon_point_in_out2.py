#!/usr/bin/python

#ref. http://stackoverflow.com/questions/11716268/point-in-polygon-algorithm
#Ray-casting algorithm (http://en.wikipedia.org/wiki/Point_in_polygon)
#with considering point on an edge of polygon.
def PointInPolygon2D2(points, point, include_on_edge=True):
  if PointOnPolygon2D(points, point):  return include_on_edge
  c= False
  j= len(points)-1
  for i in range(len(points)):
    if ((points[i][1]>point[1]) != (points[j][1]>point[1])) and (point[0] < (points[j][0]-points[i][0]) * (point[1]-points[i][1]) / (points[j][1]-points[i][1]) + points[i][0]) :
      c= not c
    j= i
  return c

#Check if point is on an edge of polygon points.
def PointOnPolygon2D(points, point, tol=1.0e-10):
  def PointOnLine(p1,p2,p):
    if (p[0]==p1[0] and p[1]==p1[1]) or (p[0]==p2[0] and p[1]==p2[1]):  return True
    if p[0]==p1[0] or p[0]==p2[0]:
      if p[0]==p1[0] and p[0]==p2[0]:
        if (p1[1]<p[1] and p[1]<p2[1]) or (p2[1]<p[1] and p[1]<p1[1]):  return True
      return False
    if abs((p[1]-p1[1])/(p[0]-p1[0])-(p2[1]-p[1])/(p2[0]-p[0]))<tol:  return True
    return False
  return PointOnLine(points[-1],points[0],point) or any(PointOnLine(p1,p2,point) for p1,p2 in zip(points[:-1],points[1:]))

def FRange(xmin,xmax,num_div):
  return [xmin+(xmax-xmin)*x/float(num_div) for x in range(num_div+1)]

if __name__=='__main__':
  def PrintEq(s):  print '%s= %r' % (s, eval(s))

  from gen_data import *
  #points= To2d(Gen3d_01())
  #points= To2d(Gen3d_02())
  #points= To2d(Gen3d_11())
  points= To2d(Gen3d_12())
  #points= To2d(Gen3d_13())

  with open('/tmp/orig.dat','w') as fp:
    for p in points:
      fp.write(' '.join(map(str,p))+'\n')

  bb_min= [min(x for x,y in points), min(y for x,y in points)]
  bb_max= [max(x for x,y in points), max(y for x,y in points)]

  def write_in_out(fp1,fp2,x,y):
    p= [x,y]
    inout= PointInPolygon2D2(points,p,include_on_edge=True)
    if inout:
      fp1.write(' '.join(map(str,p))+'\n')
    else:
      fp2.write(' '.join(map(str,p))+'\n')
    print p,inout
  with file('/tmp/points_in.dat','w') as fp1:
    with file('/tmp/points_out.dat','w') as fp2:
      for x in FRange(bb_min[0],bb_max[0],50):
        for y in FRange(bb_min[1],bb_max[1],50):
          write_in_out(fp1,fp2,x,y)
      fp1.write('\n')
      fp2.write('\n')
      for i,(x,y) in enumerate(points):
        write_in_out(fp1,fp2,x,y)
        if i>0:  write_in_out(fp1,fp2,0.5*(x+points[i-1][0]),0.5*(y+points[i-1][1]))

  print 'Plot by'
  print "qplot -x /tmp/orig.dat w l /tmp/points_in.dat /tmp/points_out.dat"

