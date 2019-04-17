#!/usr/bin/python
#\file    polygon_match.py
#\brief  Match two polygons by maximizing the overlapped area.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    May.24, 2017
import math
import numpy as np
import numpy.linalg as la
import random
from polygon_clip import ClipPolygon
from polygon_area import PolygonArea
from polygon_convexhull import ConvexHull
from scipy.optimize import minimize as scipy_minimize

#Generate a random number of uniform distribution of specified bound.
def Rand(xmin=-0.5,xmax=0.5):
  return random.random()*(xmax-xmin)+xmin

#Generate a random vector of uniform distribution; each dim has different bound.
def RandN(xmins,xmaxs):
  assert(len(xmins)==len(xmaxs))
  return [Rand(xmins[d],xmaxs[d]) for d in range(len(xmins))]

#Generate a random vector of uniform distribution; each dim has different bound.
def RandB(bounds):
  return RandN(bounds[0],bounds[1])

'''Move a polygon points along with axes so that it matches with points_ref.
  More specifically, find r such that intersection-area between
  points+r*axes and points_ref is maximized.
  NOTE: For simplicity of computation, points_ref is converted to a convex hull.
  axes: [ax1] or [ax1,ax2] where ax=[x,y].
    In case of [ax1], r is 1-dimensional, and r is 2-d for [ax1,ax2].
  bounds: bounds of r.
'''
def MatchPolygons(points, points_ref, axes, bounds, maxeval=1000):
  points_ref= ConvexHull(points_ref)
  points= np.array(points)
  axes= np.array(axes)
  def f_obj(r):
    move= np.dot(r,axes)
    points_mv= points+move
    intersection= ClipPolygon(points_mv.tolist(), points_ref)
    return -PolygonArea(intersection)
  r= [0.0]*len(axes)
  while f_obj(r)==0.0 and maxeval>0:
    r= RandB(bounds)
    maxeval-= 1
    print r,f_obj(r)
  if f_obj(r)==0.0:  return None, points
  bounds2= [[xmin,xmax] for xmin,xmax in zip(bounds[0],bounds[1])]
  res= scipy_minimize(f_obj, r, bounds=bounds2, options={'maxiter':maxeval})
  print res
  r= res['x']
  points_mv= points+np.dot(r,axes)
  return r, points_mv.tolist()

def Main():
  polygons=[
    [[0.729,0.049],[0.723,0.082],[0.702,0.125],[0.682,0.124],[0.654,0.106],[0.656,0.101],[0.647,0.081],[0.652,0.078],[0.651,0.071],[0.655,0.071],[0.673,0.031]],
    [[0.722,0.219],[0.717,0.220],[0.712,0.229],[0.693,0.235],[0.681,0.227],[0.672,0.230],[0.649,0.211],[0.637,0.213],[0.629,0.208],[0.626,0.216],[0.620,0.202],[0.616,0.203],[0.617,0.207],[0.609,0.200],[0.603,0.201],[0.601,0.191],[0.587,0.181],[0.589,0.175],[0.580,0.166],[0.585,0.133],[0.593,0.121],[0.605,0.113],[0.626,0.113],[0.645,0.121],[0.644,0.127],[0.651,0.123],[0.661,0.135],[0.669,0.134],[0.675,0.140],[0.702,0.148],[0.715,0.159],[0.717,0.150],[0.720,0.149],[0.721,0.167],[0.727,0.167],[0.730,0.195],[0.724,0.204]],
    [[0.820,0.156],[0.793,0.154],[0.812,0.154],[0.812,0.150],[0.803,0.149],[0.806,0.134],[0.802,0.139],[0.796,0.133],[0.786,0.140],[0.779,0.139],[0.772,0.131],[0.774,0.126],[0.782,0.127],[0.779,0.134],[0.789,0.130],[0.788,0.115],[0.794,0.109],[0.773,0.111],[0.769,0.124],[0.755,0.143],[0.749,0.144],[0.753,0.150],[0.750,0.153],[0.737,0.147],[0.731,0.149],[0.738,0.141],[0.722,0.144],[0.722,0.124],[0.726,0.126],[0.729,0.123],[0.725,0.118],[0.733,0.107],[0.733,0.090],[0.738,0.086],[0.738,0.077],[0.740,0.082],[0.744,0.080],[0.749,0.041],[0.757,0.039],[0.758,0.032],[0.763,0.034],[0.762,0.040],[0.769,0.037],[0.769,0.008],[0.781,0.024],[0.778,0.034],[0.788,0.043],[0.828,0.144],[0.819,0.150]],
    [[0.6,0.05],[0.65,0.05],[0.65,0.1],[0.6,0.1]],
    [[0.6,0.15],[0.6,0.2],[0.65,0.15],[0.65,0.2]],
    [[0.65,0.05],[0.7,0.05],[0.7,0.1]],
    [[0.6,0.05],[0.6,0.1],[0.65,0.05],[0.65,0.1]],
    [[0.7361939149219346, 0.29554086304276533], [0.7390000543833941, 0.3099503163934977], [0.7271370140934456, 0.34403529858804316], [0.7086499782646767, 0.34552522999470653], [0.6701051693267207, 0.33560752822065054], [0.6775118852904158, 0.2908905866198169], [0.6856281049655686, 0.27993422119568606], [0.7112291660135358, 0.2893963436516301], [0.7210350031486497, 0.28656348584059155]],
    [[0.6742816148667916, 0.3359038319949318], [0.6803237086110518, 0.2958387338776848], [0.6844753947744056, 0.28423344758258273], [0.689962535914821, 0.2801546881318294], [0.7162682404694547, 0.289925010909774], [0.7364615438257127, 0.291463048082854], [0.7425826298593776, 0.29969855936235784], [0.744559129730935, 0.30775460173666314], [0.7407489423479557, 0.32286454012087373], [0.7279985300616816, 0.3488703531207672], [0.7013981274119427, 0.3420654849830679], [0.6938717202912233, 0.3433965222819394]],
    [[0.7507708546072512, 0.25996311890044116], [0.7376167602861947, 0.2744641880682156], [0.7160805615970434, 0.27610851243886075], [0.6924527037436328, 0.2639642462596887], [0.6875804618236175, 0.2480377251283642], [0.6942749628537697, 0.23283832507923855], [0.7235939767619256, 0.22689582154928742], [0.7467562671752629, 0.2400017651022114]],
    [[0.6936233992304721, 0.23914467824135066], [0.7054265581837668, 0.23018896227621816], [0.7157636161272518, 0.2282861312931919], [0.7394732556298717, 0.2328232538861721], [0.7550731625777863, 0.24448432698139574], [0.7569515134788138, 0.26394679960205075], [0.7458742503572292, 0.278555304995346], [0.7140064731859939, 0.27789827052252536], [0.7033487528883895, 0.272402766441005], [0.6923723062666173, 0.25948147716421593], [0.6902916297733366, 0.25169667635231363]],
    ]

  ##Test:
  #polygon1= polygons[0]
  #polygon2= polygons[1]
  ##axes= [[1.0,-1.0]]
  ##bounds= [[-1],[1]]
  #axes= [[1.0,0.0],[0.0,1.0]]
  #bounds= [[-1,-1],[1,1]]

  ##Real data-1:
  #polygon1= polygons[7]
  #polygon2= polygons[8]
  #axes= [[0.777959419569747, 0.6283145243448557]]
  #bounds= [[-0.05],[0.05]]

  #Real data-2:
  polygon1= polygons[9]
  polygon2= polygons[10]
  axes= [[-0.9661088877392736, 0.2581348814693273]]
  bounds= [[-0.05],[0.05]]
  #axes= [[1.0, 0.0], [0.0, 1.0]]
  #bounds= [[-0.05,-0.05],[0.05,0.05]]

  r,polygon1m= MatchPolygons(polygon1, polygon2, axes, bounds)

  #points_ref= ConvexHull(polygon2)
  #axes= np.array(axes)
  #points= np.array(polygon1)
  #r= [-0.05]
  #points_mv= points+np.dot(r,axes)
  #intersection= ClipPolygon(points_mv.tolist(), points_ref)
  #print 'intersection=',intersection
  #print 'area=',PolygonArea(intersection)
  #polygon1m= points_mv.tolist()

  def write_polygon(fp,polygon):
    if len(polygon)>0:
      for pt in polygon+[polygon[0]]:
        fp.write('%s\n'%' '.join(map(str,pt)))
    fp.write('\n')

  fp= open('/tmp/polygons.dat','w')
  write_polygon(fp,polygon1)
  write_polygon(fp,polygon2)
  write_polygon(fp,polygon1m)
  fp.close()

def PlotGraphs():
  print 'Plotting graphs..'
  import os
  commands=[
    '''qplot -x2 aaa
        /tmp/polygons.dat u 1:2:'(column(-1)+1)' lc var w l
        &''',
        #/tmp/polygons.dat u 1:2:-1 lc var w l
    '''''',
    '''''',
    ]
  for cmd in commands:
    if cmd!='':
      cmd= ' '.join(cmd.splitlines())
      print '###',cmd
      os.system(cmd)

  print '##########################'
  print '###Press enter to close###'
  print '##########################'
  raw_input()
  os.system('qplot -x2kill aaa')

if __name__=='__main__':
  import sys
  if len(sys.argv)>1 and sys.argv[1] in ('p','plot','Plot','PLOT'):
    PlotGraphs()
    sys.exit(0)
  Main()
