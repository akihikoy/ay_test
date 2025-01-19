#!/usr/bin/python3
#\file    polygon_visible_vert.py
#\brief   Get visible vertices from a point using Ray-casting algorithm.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jul.30, 2023

import numpy as np
from line_line_intersect2 import DoLineLineIntersect
#from polygon_is_clockwise2 import PolygonIsClockwise
from polygon_point_in_out import PointInPolygon2D

#def DoLineLineIntersectExceptVert(p1, p2, pA, pB, tol=1e-8):
  #same_pt= lambda pa,pb: max(abs(pa[0]-pb[0]),abs(pa[1]-pb[1]))<tol
  #if DoLineLineIntersect(p1, p2, pA, pB):
    #if any((same_pt(p1,pA),same_pt(p1,pB),same_pt(p2,pA),same_pt(p2,pB))):
      #return False
    #return True
  #return False

#Get visible vertices from a point.
#ref. https://en.wikipedia.org/wiki/Visibility_polygon#Optimal_algorithms_for_a_point_in_a_simple_polygon
def GetVisibleVertices(points, point):
  if len(points)<3:  return None
  is_close= lambda pa,pb: np.linalg.norm(np.array(pa)-pb)<min_dist
  points= list(points)  #Convert numpy.array
  stack= [0,1]
  #for k1 in range(3):
    #print('{},dist= {}'.format(k1,np.linalg.norm(np.array(point)-points[k1])))
    #if not is_close(point,k1):  stack.append(k1)
    #if len(stack)==2:  break
  #if len(stack)!=2:  return None
  for k1 in range(stack[-1],len(points)):
    k2= k1+1 if k1+1<len(points) else 0
    p1,p2= points[k1],points[k2]
    #print('{},dist= {}'.format(k2,np.linalg.norm(np.array(point)-p2)))
    #if is_close(point,p2):  continue
    #Add p2 is it is not hidden by the polygon in stack.
    is_visible= all(not DoLineLineIntersect(points[s1],points[s2],point,p2)
                    for s1,s2 in zip(stack[:-1],stack[1:]) )
    if is_visible and k2!=0:
      stack.append(k2)
    #print(k1,k2,is_visible,stack)
    #Remove points in stack it they are hidden by p1-p2.
    stack= [s for s in stack
            if s==k1 or s==k2 or not DoLineLineIntersect(p1,p2,point,points[s])]
    #print('-----',k1,k2,is_visible,stack)
  visibility= [False]*len(points)
  for s in stack:  visibility[s]= True
  return visibility

#Get visible vertices from a vertex. Neighbor vertices are excluded.
def GetVisibleVerticesFromVertex(points, i_point):
  visibility= GetVisibleVertices(points, GetVertexPointWithTinyOffset(points, i_point))
  visibility= [visible and abs(i_point-j_point)>1 and abs(i_point-j_point)!=len(points)-1
                  for j_point,visible in enumerate(visibility)]
  return visibility

#Return a point on a polygon points[i_point] with a tiny offset
# so that the point is inside (if inside==True or outside (if inside==False) of the polygon.
def GetVertexPointWithTinyOffset(points, i_point, inside=True, r_offset=1.0e-5):
  point= np.array(points[i_point])
  p1= points[i_point-1 if i_point-1>=0 else len(points)-1]
  p2= points[i_point+1 if i_point+1<len(points) else 0]
  dp= r_offset*((p1-point)+(p2-point))
  if inside:
    point= point+dp if PointInPolygon2D(points, point+dp) else point-dp
  else:
    point= point+dp if not PointInPolygon2D(points, point+dp) else point-dp
  return point

def Main():
  polygons=[
    [[0.744, 0.54], [0.532, 1.124], [1.12, 0.996], [1.324, 1.432], [1.608, 1.12], [2.04, 0.632], [1.464, 0.696], [1.224, 0.328], [1.16, 0.7]],
    [[0.784, 0.58], [1.172, 1.096], [1.72, 0.524], [1.724, 1.496], [0.84, 1.516]],
    [[0.704, 0.748], [1.18, 0.42], [1.516, 0.704], [1.416, 0.936], [1.028, 1.052], [0.876, 1.384], [1.476, 1.28], [1.932, 1.012], [2.056, 1.436], [1.172, 1.752], [0.48, 1.62], [0.364, 1.064]],
    [[0.556, 0.764], [0.952, 0.316], [1.988, 0.368], [2.152, 0.668], [2.012, 0.928], [1.484, 0.872], [1.228, 0.936], [1.2, 1.156], [1.36, 1.312], [1.952, 1.328], [2.24, 1.432], [2.1, 1.648], [1.632, 1.844], [1.016, 1.812], [0.552, 1.492], [0.776, 1.324], [0.812, 1.068], [0.524, 0.952]],
    [[0.504, 0.724], [0.908, 0.296], [1.7, 0.304], [1.996, 0.668], [1.98, 0.872], [1.8, 0.996], [1.396, 0.984], [1.232, 1.296], [1.732, 1.484], [1.892, 1.792], [1.088, 1.756], [0.396, 1.5], [0.684, 1.26], [0.728, 0.98]],
    [[0.729,0.049],[0.723,0.082],[0.702,0.125],[0.682,0.124],[0.654,0.106],[0.656,0.101],[0.647,0.081],[0.652,0.078],[0.651,0.071],[0.655,0.071],[0.673,0.031]],
    [[0.722,0.219],[0.717,0.220],[0.712,0.229],[0.693,0.235],[0.681,0.227],[0.672,0.230],[0.649,0.211],[0.637,0.213],[0.629,0.208],[0.626,0.216],[0.620,0.202],[0.616,0.203],[0.617,0.207],[0.609,0.200],[0.603,0.201],[0.601,0.191],[0.587,0.181],[0.589,0.175],[0.580,0.166],[0.585,0.133],[0.593,0.121],[0.605,0.113],[0.626,0.113],[0.645,0.121],[0.644,0.127],[0.651,0.123],[0.661,0.135],[0.669,0.134],[0.675,0.140],[0.702,0.148],[0.715,0.159],[0.717,0.150],[0.720,0.149],[0.721,0.167],[0.727,0.167],[0.730,0.195],[0.724,0.204]],
    [[0.820,0.156],[0.793,0.154],[0.812,0.154],[0.812,0.150],[0.803,0.149],[0.806,0.134],[0.802,0.139],[0.796,0.133],[0.786,0.140],[0.779,0.139],[0.772,0.131],[0.774,0.126],[0.782,0.127],[0.779,0.134],[0.789,0.130],[0.788,0.115],[0.794,0.109],[0.773,0.111],[0.769,0.124],[0.755,0.143],[0.749,0.144],[0.753,0.150],[0.750,0.153],[0.737,0.147],[0.731,0.149],[0.738,0.141],[0.722,0.144],[0.722,0.124],[0.726,0.126],[0.729,0.123],[0.725,0.118],[0.733,0.107],[0.733,0.090],[0.738,0.086],[0.738,0.077],[0.740,0.082],[0.744,0.080],[0.749,0.041],[0.757,0.039],[0.758,0.032],[0.763,0.034],[0.762,0.040],[0.769,0.037],[0.769,0.008],[0.781,0.024],[0.778,0.034],[0.788,0.043],[0.828,0.144],[0.819,0.150]],
    #[[0.6,0.05],[0.65,0.05],[0.65,0.1],[0.6,0.1]],
    #[[0.6,0.15],[0.6,0.2],[0.65,0.15],[0.65,0.2]],
    #[[0.65,0.05],[0.7,0.05],[0.7,0.1]],
    #[[0.6,0.05],[0.6,0.1],[0.65,0.05],[0.65,0.1]],
    ]

  i_poly= np.random.choice(list(range(len(polygons))))
  if np.random.uniform(0,1)<0.5:
    i_point= np.random.choice(list(range(len(polygons[i_poly]))))
    point= GetVertexPointWithTinyOffset(polygons[i_poly], i_point)
    print('Vertex point {} is selected: {}'.format(i_point,point))
  else:
    bb_min,bb_max= np.min(polygons[i_poly],axis=0),np.max(polygons[i_poly],axis=0)
    point= np.random.uniform(bb_min,bb_max)
    print('Random point is selected: {}'.format(point))
  visibility= GetVisibleVertices(polygons[i_poly], point)
  print('visible vertices=',np.array(list(range(len(polygons[i_poly]))))[visibility])

  def write_polygon(fp,polygon):
    if len(polygon)>0:
      for pt in polygon+[polygon[0]]:
        fp.write('%s\n'%' '.join(map(str,pt)))
    fp.write('\n')

  with open('/tmp/polygons.dat','w') as fp:
    for polygon in [polygons[i_poly]]:
      write_polygon(fp,polygon)
  with open('/tmp/visibility.dat','w') as fp:
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
  input()
  os.system('qplot -x2kill aaa')

if __name__=='__main__':
  import sys
  if len(sys.argv)>1 and sys.argv[1] in ('p','plot','Plot','PLOT'):
    PlotGraphs()
    sys.exit(0)
  Main()

  PlotGraphs()
  sys.exit(0)
