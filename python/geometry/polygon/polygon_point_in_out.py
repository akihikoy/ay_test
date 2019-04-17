#!/usr/bin/python

#ref. http://stackoverflow.com/questions/11716268/point-in-polygon-algorithm
#Ray-casting algorithm (http://en.wikipedia.org/wiki/Point_in_polygon)
def PointInPolygon2D(points, point):
  c= False
  j= len(points)-1
  for i in range(len(points)):
    if ((points[i][1]>point[1]) != (points[j][1]>point[1])) and (point[0] < (points[j][0]-points[i][0]) * (point[1]-points[i][1]) / (points[j][1]-points[i][1]) + points[i][0]) :
      c= not c
    j= i
  return c

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

  fp= file('/tmp/orig.dat','w')
  for p in points:
    fp.write(' '.join(map(str,p))+'\n')
  fp.close()

  bb_min= [min(x for x,y in points), min(y for x,y in points)]
  bb_max= [max(x for x,y in points), max(y for x,y in points)]

  fp1= file('/tmp/points_in.dat','w')
  fp2= file('/tmp/points_out.dat','w')
  for x in FRange(bb_min[0],bb_max[0],50):
    for y in FRange(bb_min[1],bb_max[1],50):
      p= [x,y]
      inout= PointInPolygon2D(points,p)
      if inout:
        fp1.write(' '.join(map(str,p))+'\n')
      else:
        fp2.write(' '.join(map(str,p))+'\n')
      print p,inout
  fp1.close()
  fp2.close()

  print 'Plot by'
  print "qplot -x /tmp/orig.dat w l /tmp/points_in.dat /tmp/points_out.dat"

