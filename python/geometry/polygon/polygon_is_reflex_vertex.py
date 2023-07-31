#!/usr/bin/python
#\file    polygon_is_reflex_vertex.py
#\brief   Check if a vertex is reflex vertex (angle>180).
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jul.30, 2023
from __future__ import print_function
import numpy as np
from polygon_is_clockwise2 import PolygonIsClockwise
from polygon_visible_vert import GetVisibleVertices, GetVertexPointWithTinyOffset
from geometry import GetAngle2

#Check if a vertex i_point (index) is reflex vertex (angle>180).
#  NOTE: Assume that polygon is sorted clockwise.
def PolygonIsReflexVertex(polygon, i_point):
  p0= np.array(polygon[i_point])
  p1= polygon[i_point-1 if i_point-1>=0 else len(polygon)-1]
  p2= polygon[i_point+1 if i_point+1<len(polygon) else 0]
  theta= GetAngle2(p1-p0, p2-p0)
  #print('i_point={}: theta={}'.format(i_point,theta))
  return theta<0.0

def Main():
  polygons=[
    [[0.744, 0.54], [0.532, 1.124], [1.12, 0.996], [1.324, 1.432], [1.608, 1.12], [2.04, 0.632], [1.464, 0.696], [1.224, 0.328], [1.16, 0.7]],
    [[0.784, 0.58], [1.172, 1.096], [1.72, 0.524], [1.724, 1.496], [0.84, 1.516]],
    [[0.704, 0.748], [1.18, 0.42], [1.516, 0.704], [1.416, 0.936], [1.028, 1.052], [0.876, 1.384], [1.476, 1.28], [1.932, 1.012], [2.056, 1.436], [1.172, 1.752], [0.48, 1.62], [0.364, 1.064]],
    [[0.729,0.049],[0.723,0.082],[0.702,0.125],[0.682,0.124],[0.654,0.106],[0.656,0.101],[0.647,0.081],[0.652,0.078],[0.651,0.071],[0.655,0.071],[0.673,0.031]],
    [[0.722,0.219],[0.717,0.220],[0.712,0.229],[0.693,0.235],[0.681,0.227],[0.672,0.230],[0.649,0.211],[0.637,0.213],[0.629,0.208],[0.626,0.216],[0.620,0.202],[0.616,0.203],[0.617,0.207],[0.609,0.200],[0.603,0.201],[0.601,0.191],[0.587,0.181],[0.589,0.175],[0.580,0.166],[0.585,0.133],[0.593,0.121],[0.605,0.113],[0.626,0.113],[0.645,0.121],[0.644,0.127],[0.651,0.123],[0.661,0.135],[0.669,0.134],[0.675,0.140],[0.702,0.148],[0.715,0.159],[0.717,0.150],[0.720,0.149],[0.721,0.167],[0.727,0.167],[0.730,0.195],[0.724,0.204]],
    [[0.820,0.156],[0.793,0.154],[0.812,0.154],[0.812,0.150],[0.803,0.149],[0.806,0.134],[0.802,0.139],[0.796,0.133],[0.786,0.140],[0.779,0.139],[0.772,0.131],[0.774,0.126],[0.782,0.127],[0.779,0.134],[0.789,0.130],[0.788,0.115],[0.794,0.109],[0.773,0.111],[0.769,0.124],[0.755,0.143],[0.749,0.144],[0.753,0.150],[0.750,0.153],[0.737,0.147],[0.731,0.149],[0.738,0.141],[0.722,0.144],[0.722,0.124],[0.726,0.126],[0.729,0.123],[0.725,0.118],[0.733,0.107],[0.733,0.090],[0.738,0.086],[0.738,0.077],[0.740,0.082],[0.744,0.080],[0.749,0.041],[0.757,0.039],[0.758,0.032],[0.763,0.034],[0.762,0.040],[0.769,0.037],[0.769,0.008],[0.781,0.024],[0.778,0.034],[0.788,0.043],[0.828,0.144],[0.819,0.150]],
    #[[0.6,0.05],[0.65,0.05],[0.65,0.1],[0.6,0.1]],
    #[[0.6,0.15],[0.6,0.2],[0.65,0.15],[0.65,0.2]],
    #[[0.65,0.05],[0.7,0.05],[0.7,0.1]],
    #[[0.6,0.05],[0.6,0.1],[0.65,0.05],[0.65,0.1]],
    ]

  i_poly= np.random.choice(list(range(len(polygons))))
  if not PolygonIsClockwise(polygons[i_poly]):  polygons[i_poly].reverse()
  reflex_vertices= []
  for i_point in range(len(polygons[i_poly])):
    if PolygonIsReflexVertex(polygons[i_poly], i_point):
      print('i_point={}: IsReflexVertex={}'.format(i_point,True))
      point= GetVertexPointWithTinyOffset(polygons[i_poly], i_point)
      visibility= GetVisibleVertices(polygons[i_poly], point)
      print('--visible vertices=',np.array(range(len(polygons[i_poly])))[visibility])
      reflex_vertices.append((i_point,point,visibility))
  print('reflex_vertices=',[i_point for i_point,point,visibility in reflex_vertices])

  def write_polygon(fp,polygon):
    if len(polygon)>0:
      for pt in polygon+[polygon[0]]:
        fp.write('%s\n'%' '.join(map(str,pt)))
    fp.write('\n')

  with open('/tmp/polygons.dat','w') as fp:
    for polygon in [polygons[i_poly]]:
      write_polygon(fp,polygon)
  with open('/tmp/reflex_vertices.dat','w') as fp:
    write_polygon(fp,[polygons[i_poly][i_point] for i_point,point,visibility in reflex_vertices])
  with open('/tmp/visibility.dat','w') as fp:
    for i_point,point,visibility in reflex_vertices:
      for idx,pt in enumerate(polygons[i_poly]):
        if visibility[idx]:
          fp.write('%s\n'%' '.join(map(str,point)))
          fp.write('%s\n'%' '.join(map(str,pt)))
          fp.write('\n')

def PlotGraphs():
  print('Plotting graphs..')
  import os
  commands=[
    '''qplot -x2 aaa
        /tmp/polygons.dat u 1:2 w l lw 3
        /tmp/visibility.dat u 1:2 w l
        /tmp/reflex_vertices.dat u 1:2 w l lw 3 lt 5
        &''',
        #/tmp/polygons.dat u 1:2:-1 lc var w l
    '''''',
    '''''',
    ]
  for cmd in commands:
    if cmd!='':
      cmd= ' '.join(cmd.splitlines())
      print('###',cmd)
      os.system(cmd)

  print('##########################')
  print('###Press enter to close###')
  print('##########################')
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
