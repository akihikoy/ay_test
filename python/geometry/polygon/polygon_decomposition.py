#!/usr/bin/python
#\file    polygon_decomposition.py
#\brief   Test code of polygon decomposition by Mark Keil's algorithm.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jul.30, 2023
from __future__ import print_function
from polygon_is_clockwise2 import PolygonIsClockwise
from polygon_visible_vert import GetVisibleVertices, GetVertexPointWithTinyOffset
from polygon_is_reflex_vertex import PolygonIsReflexVertex
import numpy as np

#ref. https://mpen.ca/406/keil

def DecomposePolygon(polygon):
  polygon= list(polygon)  #Convert numpy.array
  is_reversed= False
  if not PolygonIsClockwise(polygon):
    is_reversed= True
    polygon.reverse()
  visibilities= [GetVisibleVertices(polygon, GetVertexPointWithTinyOffset(polygon, i_point)) for i_point in range(len(polygon))]

  def decom(idxes_poly, depth=0):
    ndiags= None
    min_diags= []
    if len(idxes_poly)<3:  return []
    #if depth>=3  :return []  #TEST:Does this speed up?
    #print('{} debug'.format(depth),len(idxes_poly))
    sub_poly= [polygon[i_point] for i_point in idxes_poly]
    is_reflex_vertex= {i_point: PolygonIsReflexVertex(sub_poly, i_local) for i_local,i_point in enumerate(idxes_poly)}
    for i_point in idxes_poly:
      if not is_reflex_vertex[i_point]:  continue
      visibility= visibilities[i_point]
      for j_point in idxes_poly:
        if not visibility[j_point]:  continue
        #if not is_reflex_vertex[j_point]:  continue  #TEST:Does this speed up?
        if i_point==j_point:  continue
        i_pos,j_pos= idxes_poly.index(i_point),idxes_poly.index(j_point)
        if abs(i_pos-j_pos)==1:  continue
        if i_pos>j_pos:  i_pos,j_pos= j_pos,i_pos
        idxes_tmp1= idxes_poly[i_pos:j_pos+1]
        idxes_tmp2= idxes_poly[j_pos:]+idxes_poly[:i_pos+1]
        if len(idxes_tmp1)<3 or len(idxes_tmp2)<3:  continue
        tmp1= decom(idxes_tmp1, depth+1)
        tmp2= decom(idxes_tmp2, depth+1)
        tmp1= tmp1+tmp2
        if ndiags is None or len(tmp1)<ndiags:
          min_diags= tmp1+[[i_point,j_point]]
          ndiags= len(tmp1)
          print('min_diags= {} (ndiags={})'.format(min_diags,ndiags))
          #print('--visibility= {}'.format([visibilities[ip][jp] for ip,jp in min_diags]))
    return min_diags
  min_diags= decom(list(range(len(polygon))))
  print('[]min_diags= {}'.format(min_diags))
  #print('[]--visibility= {}'.format([visibilities[ip][jp] for ip,jp in min_diags]))
  if is_reversed:
    l_poly= len(polygon)
    min_diags= [(l_poly-ip-1, l_poly-jp-1) for ip,jp in min_diags]
  return min_diags


def Main():
  import time
  polygons=[
    [[0.744, 0.54], [0.532, 1.124], [1.12, 0.996], [1.324, 1.432], [1.608, 1.12], [2.04, 0.632], [1.464, 0.696], [1.224, 0.328], [1.16, 0.7]],
    [[0.784, 0.58], [1.172, 1.096], [1.72, 0.524], [1.724, 1.496], [0.84, 1.516]],
    [[0.704, 0.748], [1.18, 0.42], [1.516, 0.704], [1.416, 0.936], [1.028, 1.052], [0.876, 1.384], [1.476, 1.28], [1.932, 1.012], [2.056, 1.436], [1.172, 1.752], [0.48, 1.62], [0.364, 1.064]],
    [[0.729,0.049],[0.723,0.082],[0.702,0.125],[0.682,0.124],[0.654,0.106],[0.656,0.101],[0.647,0.081],[0.652,0.078],[0.651,0.071],[0.655,0.071],[0.673,0.031]],
    #[[0.722,0.219],[0.717,0.220],[0.712,0.229],[0.693,0.235],[0.681,0.227],[0.672,0.230],[0.649,0.211],[0.637,0.213],[0.629,0.208],[0.626,0.216],[0.620,0.202],[0.616,0.203],[0.617,0.207],[0.609,0.200],[0.603,0.201],[0.601,0.191],[0.587,0.181],[0.589,0.175],[0.580,0.166],[0.585,0.133],[0.593,0.121],[0.605,0.113],[0.626,0.113],[0.645,0.121],[0.644,0.127],[0.651,0.123],[0.661,0.135],[0.669,0.134],[0.675,0.140],[0.702,0.148],[0.715,0.159],[0.717,0.150],[0.720,0.149],[0.721,0.167],[0.727,0.167],[0.730,0.195],[0.724,0.204]],
    #[[0.820,0.156],[0.793,0.154],[0.812,0.154],[0.812,0.150],[0.803,0.149],[0.806,0.134],[0.802,0.139],[0.796,0.133],[0.786,0.140],[0.779,0.139],[0.772,0.131],[0.774,0.126],[0.782,0.127],[0.779,0.134],[0.789,0.130],[0.788,0.115],[0.794,0.109],[0.773,0.111],[0.769,0.124],[0.755,0.143],[0.749,0.144],[0.753,0.150],[0.750,0.153],[0.737,0.147],[0.731,0.149],[0.738,0.141],[0.722,0.144],[0.722,0.124],[0.726,0.126],[0.729,0.123],[0.725,0.118],[0.733,0.107],[0.733,0.090],[0.738,0.086],[0.738,0.077],[0.740,0.082],[0.744,0.080],[0.749,0.041],[0.757,0.039],[0.758,0.032],[0.763,0.034],[0.762,0.040],[0.769,0.037],[0.769,0.008],[0.781,0.024],[0.778,0.034],[0.788,0.043],[0.828,0.144],[0.819,0.150]],
    #[[0.6,0.05],[0.65,0.05],[0.65,0.1],[0.6,0.1]],
    #[[0.6,0.15],[0.6,0.2],[0.65,0.15],[0.65,0.2]],
    #[[0.65,0.05],[0.7,0.05],[0.7,0.1]],
    #[[0.6,0.05],[0.6,0.1],[0.65,0.05],[0.65,0.1]],
    ]

  i_poly= np.random.choice(list(range(len(polygons))))
  t_start= time.time()
  diags= DecomposePolygon(polygons[i_poly])
  t_end= time.time()
  print('DecomposePolygon=',diags)
  print('compt. time={}s for # of vertex={}'.format(t_end-t_start,len(polygons[i_poly])))

  def write_polygon(fp,polygon):
    if len(polygon)>0:
      for pt in polygon+[polygon[0]]:
        fp.write('%s\n'%' '.join(map(str,pt)))
    fp.write('\n')

  with open('/tmp/polygons.dat','w') as fp:
    for polygon in [polygons[i_poly]]:
      write_polygon(fp,polygon)
  with open('/tmp/diags.dat','w') as fp:
    for i_point,j_point in diags:
      fp.write('%s\n'%' '.join(map(str,polygons[i_poly][i_point])))
      fp.write('%s\n'%' '.join(map(str,polygons[i_poly][j_point])))
      fp.write('\n')

def PlotGraphs():
  print('Plotting graphs..')
  import os
  commands=[
    '''qplot -x2 aaa
        /tmp/polygons.dat u 1:2 w l lw 3
        /tmp/diags.dat u 1:2 w l
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
