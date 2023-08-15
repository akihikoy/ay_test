#!/usr/bin/python
#\file    polygon_split_at_reflex.py
#\brief   Split a polygon into two at a reflex vertex so that the sum of convex-ratio is maximized.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jul.31, 2023
from __future__ import print_function
from polygon_is_clockwise2 import PolygonIsClockwise
from polygon_convex_ratio import ConvexRatio
from polygon_is_reflex_vertex import PolygonIsReflexVertex
from polygon_visible_vert import GetVisibleVerticesFromVertex
import numpy as np
import time

#Split a polygon into two at a reflex vertex so that the convex-ratio is maximized.
#  Return: [poly1,poly2].
#  If there is no reflex vertex, or the number of vertices after split is less than 3, return [points].
def SplitPolygonAtReflexVertex(points, num_approx=None):
  def split(poly,i,j):
    if i>j:  i,j= j,i
    return [poly[i:j+1], poly[j:]+poly[:i+1]]
  #t0= time.time()
  #print('# points={}'.format(len(points)))
  if num_approx is not None and len(points)>num_approx:
    idxs_map= np.round(np.linspace(0,len(points)-1,num_approx)).astype(int)
    points_orig= points
    points= [points[idx] for idx in idxs_map]
  else:
    idxs_map= None
  if not PolygonIsClockwise(points):  points.reverse()
  #print('  t1= {:6.2f}ms'.format((time.time()-t0)*1.e3)); t0=time.time()
  visibilities= [GetVisibleVerticesFromVertex(points, i_point) for i_point in range(len(points))]
  #print('  t2= {:6.2f}ms'.format((time.time()-t0)*1.e3)); t0=time.time()
  #print('visibilities=',{i_point:np.array(range(len(points)))[visibility] for i_point,visibility in enumerate(visibilities)})
  idxs_reflex= [i_point for i_point in range(len(points)) if PolygonIsReflexVertex(points, i_point)]
  #print('  t3= {:6.2f}ms'.format((time.time()-t0)*1.e3)); t0=time.time()
  #print('idxs_reflex=',idxs_reflex)
  if len(idxs_reflex)==0:  return [points]
  except_mask= lambda array,idx: [i==idx or not visibilities[idx][i] for i in range(len(array))]
  idxs_shortest= [np.argmin(np.ma.array(np.linalg.norm(np.array(points)-points[i_reflex],axis=1),
                                        mask=except_mask(points,i_reflex)))
                  for i_reflex in idxs_reflex]
  #print('  t4= {:6.2f}ms'.format((time.time()-t0)*1.e3)); t0=time.time()
  #print('idxs_shortest=',idxs_shortest)
  polygons_split= [split(points,i_reflex,i_shortest)
                   for i_reflex,i_shortest in zip(idxs_reflex,idxs_shortest)]
  #print('  t5= {:6.2f}ms'.format((time.time()-t0)*1.e3)); t0=time.time()
  convex_ratio= [(ConvexRatio(poly1),ConvexRatio(poly2))
                 if len(poly1)>=3 and len(poly2)>=3 else (0.0,0.0)
                 for poly1,poly2 in polygons_split]
  #print('  t6= {:6.2f}ms'.format((time.time()-t0)*1.e3)); t0=time.time()
  #print('convex_ratio=',convex_ratio)
  i_best= np.argmax([min(cr1,cr2) for cr1,cr2 in convex_ratio])
  #poly1,poly2= polygons_split[i_best]
  if idxs_map is None:
    poly1,poly2= split(points,idxs_reflex[i_best],idxs_shortest[i_best])
  else:
    points= points_orig
    poly1,poly2= split(points,idxs_map[idxs_reflex[i_best]],idxs_map[idxs_shortest[i_best]])
  #print('  t7= {:6.2f}ms'.format((time.time()-t0)*1.e3)); t0=time.time()
  #print('poly1,poly2=',poly1,poly2)
  if len(poly1)<3 or len(poly2)<3:  return [points]  #No split is better.
  return [poly1,poly2]


def Main():
  import time
  def write_polygon(fp,polygon):
    if len(polygon)>0:
      for pt in polygon+[polygon[0]]:
        fp.write('%s\n'%' '.join(map(str,pt)))
    fp.write('\n')

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

  polygon= polygons[np.random.choice(list(range(len(polygons))))]
  t_start= time.time()
  sub_polys= SplitPolygonAtReflexVertex(polygon, num_approx=None)
  t_end= time.time()
  print('# of sub polygons:',len(sub_polys))
  print('compt. time={}s for # of vertex={}'.format(t_end-t_start,len(polygon)))
  #print(sub_polys)

  with open('/tmp/polygons.dat','w') as fp:
    write_polygon(fp,polygon)
  with open('/tmp/sub_poly0.dat','w') as fp:
    if len(sub_polys)>0:  write_polygon(fp,sub_polys[0])
  with open('/tmp/sub_poly1.dat','w') as fp:
    if len(sub_polys)>1:  write_polygon(fp,sub_polys[1])

def PlotGraphs():
  print('Plotting graphs..')
  import os
  commands=[
    '''qplot -x2 aaa
        /tmp/polygons.dat u 1:2:'(column(-1)+1)' w lp lc var pt 4
        /tmp/sub_poly0.dat w lp pt 4
        /tmp/sub_poly1.dat w lp pt 4
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
