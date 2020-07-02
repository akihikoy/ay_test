#!/usr/bin/python
#\file    polygon_overlap.py
#\brief   Check if two polygons are overlapping.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jul.01, 2020
from polygon_point_in_out import PointInPolygon2D

#Check if polygons points1 and points2 are overlapping.
#  WARNING: This algorithm is incomplete.
#  Even when no point is included in the other polygon, they may have overlap.
#  WARNING: This algorithm does not handle a point on an edge of polygon correctly for compt cost.
#  For that purpose, use PointInPolygon2D2 with include_on_edge=True/False.
def PolygonOverlap(points1, points2):
  if len(points1)<len(points2):
    return PointInPolygon2D(points2, points1[0]) or any(PointInPolygon2D(points1, p2) for p2 in points2)
  else:
    return PointInPolygon2D(points1, points2[0]) or any(PointInPolygon2D(points2, p1) for p1 in points1)

#Check if polygon points_outer includes points_inner.
def PolygonInclude(points_outer, points_inner):
  return all(PointInPolygon2D(points_outer, pi) for pi in points_inner)


if __name__=='__main__':
  from gen_data import *
  polygons=[
    lambda:To2d(Gen3d_01()),
    lambda:To2d(Gen3d_02()),
    lambda:To2d(Gen3d_11()),
    lambda:To2d(Gen3d_12()),
    lambda:To2d(Gen3d_13())]
  #points1,points2= polygons[0](),polygons[1]()
  #points1,points2= polygons[2](),polygons[3]()
  #points1,points2= polygons[3](),polygons[4]()
  #points1,points2= polygons[0](),polygons[2]()
  #points1,points2= polygons[2](),polygons[4]()
  points1,points2= polygons[2](),polygons[3](); points2= [[x+1.,y] for [x,y] in points2]
  #points1,points2= polygons[2](),polygons[3](); points2= [[x+1.,y+1.] for [x,y] in points2]

  print 'overlapping, points1<points2, points2<points1='
  print PolygonOverlap(points1,points2), PolygonInclude(points2,points1), PolygonInclude(points1,points2)


  with open('/tmp/points1.dat','w') as fp:
    for p in points1:
      fp.write(' '.join(map(str,p))+'\n')
  with open('/tmp/points2.dat','w') as fp:
    for p in points2:
      fp.write(' '.join(map(str,p))+'\n')
  print 'qplot -x /tmp/points1.dat w l /tmp/points2.dat w l'

