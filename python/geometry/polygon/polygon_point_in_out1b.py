#!/usr/bin/python3

from polygon_point_in_out import PointInPolygon2D as PointInPolygon2D_1


from shapely import geometry
def PointInPolygon2D_2(points, point):
  line= geometry.LineString(points)
  pt= geometry.Point(point)
  polygon= geometry.Polygon(line)
  return polygon.contains(pt)


from scipy.spatial import ConvexHull
#hull= ConvexHull(points)
def PointInPolygon2D_3(hull, point, tol=1e-12):
  return all(
      (np.dot(eq[:-1], point) + eq[-1] <= tol)
      for eq in hull.equations)

from scipy.spatial import Delaunay
#hull= Delaunay(points)
def PointInPolygon2D_4(hull, point):
  return hull.find_simplex(point)>=0


def FRange(xmin,xmax,num_div):
  return [xmin+(xmax-xmin)*x/float(num_div) for x in range(num_div+1)]

if __name__=='__main__':
  def PrintEq(s):  print('%s= %r' % (s, eval(s)))

  import time
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

  eval_points= [[x,y] for x in FRange(bb_min[0],bb_max[0],50) for y in FRange(bb_min[1],bb_max[1],50)]

  t_start= time.time()
  inout1= [PointInPolygon2D_1(points,p) for p in eval_points]
  print('PointInPolygon2D_1(ver.1) compt. time:',time.time()-t_start)

  t_start= time.time()
  inout2= [PointInPolygon2D_2(points,p) for p in eval_points]
  print('PointInPolygon2D_2(w shapely) compt. time:',time.time()-t_start)

  t_start= time.time()
  hull= ConvexHull(points)
  print('ConvexHull compt. time:',time.time()-t_start)
  inout3= [PointInPolygon2D_3(hull,p) for p in eval_points]
  print('PointInPolygon2D_3(w convexhull) compt. time:',time.time()-t_start)

  t_start= time.time()
  hull= Delaunay(points)
  print('Delaunay compt. time:',time.time()-t_start)
  #inout4= map(lambda p:PointInPolygon2D_4(hull,p), eval_points)
  inout4= PointInPolygon2D_4(hull,eval_points)
  print('PointInPolygon2D_4(w Delaunay) compt. time:',time.time()-t_start)


  with open('/tmp/points_in1.dat','w') as fp1:
    with open('/tmp/points_out1.dat','w') as fp2:
      for p,io in zip(eval_points,inout1):
        if io:
          fp1.write(' '.join(map(str,p))+'\n')
        else:
          fp2.write(' '.join(map(str,p))+'\n')

  with open('/tmp/points_in2.dat','w') as fp1:
    with open('/tmp/points_out2.dat','w') as fp2:
      for p,io in zip(eval_points,inout2):
        if io:
          fp1.write(' '.join(map(str,p))+'\n')
        else:
          fp2.write(' '.join(map(str,p))+'\n')

  with open('/tmp/points_in3.dat','w') as fp1:
    with open('/tmp/points_out3.dat','w') as fp2:
      for p,io in zip(eval_points,inout3):
        if io:
          fp1.write(' '.join(map(str,p))+'\n')
        else:
          fp2.write(' '.join(map(str,p))+'\n')

  with open('/tmp/points_in4.dat','w') as fp1:
    with open('/tmp/points_out4.dat','w') as fp2:
      for p,io in zip(eval_points,inout4):
        if io:
          fp1.write(' '.join(map(str,p))+'\n')
        else:
          fp2.write(' '.join(map(str,p))+'\n')

  print('Plot by')
  print("qplot -x /tmp/orig.dat w l /tmp/points_in1.dat /tmp/points_out1.dat")
  print("qplot -x /tmp/orig.dat w l /tmp/points_in2.dat /tmp/points_out2.dat")
  print("qplot -x /tmp/orig.dat w l /tmp/points_in3.dat /tmp/points_out3.dat")
  print("qplot -x /tmp/orig.dat w l /tmp/points_in4.dat /tmp/points_out4.dat")

