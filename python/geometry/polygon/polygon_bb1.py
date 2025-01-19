#!/usr/bin/python3
from numpy import *
import math
import bound_box1.qhull_2d as qhull_2d
import bound_box1.min_bounding_rect as min_bounding_rect

#ref. https://github.com/dbworth/minimum-area-bounding-rectangle
class TPolygonBoundBox2D:
  def __init__(self,points):
    # Find convex hull
    self.HullPoints= qhull_2d.qhull2D(points)
    # Reverse order of points, to match output from other qhull implementations
    self.HullPoints= self.HullPoints[::-1]
    # Find minimum area bounding rectangle
    self.Angle, self.Area, self.Width, self.Height, self.Center, self.CornerPoints= min_bounding_rect.minBoundingRect(self.HullPoints)

if __name__=='__main__':
  def PrintEq(s):  print('%s= %r' % (s, eval(s)))

  from gen_data import *
  #points= To2d(Gen3d_01())
  #points= To2d(Gen3d_02())
  #points= To2d(Gen3d_11())
  #points= To2d(Gen3d_12())
  points= To2d(Gen3d_13())

  bb= TPolygonBoundBox2D(points)

  print('Convex hull points: \n', bb.HullPoints, "\n")
  print("Minimum area bounding box:")
  print("  Rotation angle:", bb.Angle, "rad  (", bb.Angle*(180/math.pi), "deg )")
  print("  Width:", bb.Width, " Height:", bb.Height, "  Area:", bb.Area)
  print("  Center point: \n", bb.Center)
  print("  Corner points: \n", bb.CornerPoints)

  fp= open('/tmp/orig.dat','w')
  for p in points:
    fp.write(' '.join(map(str,p))+'\n')
  fp.close()

  fp= open('/tmp/qhull.dat','w')
  for p in bb.HullPoints:
    fp.write(' '.join(map(str,p))+'\n')
  fp.write(' '.join(map(str,bb.HullPoints[0]))+'\n')
  fp.close()

  fp= open('/tmp/corner.dat','w')
  for p in bb.CornerPoints:
    fp.write(' '.join(map(str,p))+'\n')
  fp.write(' '.join(map(str,bb.CornerPoints[0]))+'\n')
  fp.close()

  fp= open('/tmp/center.dat','w')
  fp.write(' '.join(map(str,bb.Center))+'\n')
  fp.close()

  print('Plot by')
  print("qplot -x -s 'set size ratio -1' /tmp/orig.dat w l /tmp/qhull.dat w l /tmp/corner.dat w l /tmp/center.dat w p pt 6 ps 2")
