#!/usr/bin/python3
#\file    polygon_com_2d.py
#\brief   Polygon centroid in 2D.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Mar.18, 2024
import numpy as np
from polygon_com2 import PolygonCentroid2 as PolygonCentroid3D

#Centroid of a polygon in 2D
#ref. http://en.wikipedia.org/wiki/Centroid
def PolygonCentroid2D(points):
  if len(points)==0:  return None
  if len(points)==1:  return points[0]
  assert(len(points[0])==2)
  xy= np.array(points)
  xy= np.vstack((xy,[xy[0]]))  #Extend so that xy[N]==xy[0]
  # Area calculation
  A= 0.5 * np.sum(xy[:-1, 0] * xy[1:, 1] - xy[1:, 0] * xy[:-1, 1])
  # Centroid calculations
  Cx= np.sum((xy[:-1, 0] + xy[1:, 0]) * (xy[:-1, 0] * xy[1:, 1] - xy[1:, 0] * xy[:-1, 1])) / (6. * A)
  Cy= np.sum((xy[:-1, 1] + xy[1:, 1]) * (xy[:-1, 0] * xy[1:, 1] - xy[1:, 0] * xy[:-1, 1])) / (6. * A)
  return [Cx,Cy]

if __name__=='__main__':
  from gen_data import *
  import time
  polygons=[
    [[469, 232], [464, 232], [461, 230], [461, 222], [458, 222], [457, 219], [451, 219], [450, 221], [448, 221], [448, 226], [444, 228], [441, 224], [441, 219], [422, 217], [421, 211], [423, 209], [424, 200], [423, 190], [417, 190], [416, 198], [413, 200], [412, 214], [406, 218], [390, 216], [387, 214], [364, 213], [358, 210], [346, 211], [345, 208], [320, 207], [320, 217], [316, 221], [316, 225], [318, 226], [318, 230], [322, 232], [316, 234], [316, 242], [318, 243], [318, 245], [316, 247], [316, 252], [312, 253], [312, 258], [316, 259], [316, 267], [320, 270], [320, 273], [328, 274], [329, 271], [333, 271], [334, 275], [377, 275], [394, 279], [401, 278], [414, 281], [429, 281], [429, 279], [433, 276], [434, 281], [451, 283], [452, 281], [457, 281], [459, 279], [459, 275], [463, 273], [463, 266], [459, 266], [456, 269], [453, 264], [457, 263], [458, 261], [459, 263], [463, 263], [463, 254], [465, 252], [465, 248], [467, 247], [467, 242], [465, 241], [469, 240]],
    [[212, 311], [212, 315], [214, 317], [214, 326], [216, 327], [212, 329], [212, 337], [214, 338], [216, 350], [219, 351], [233, 351], [234, 342], [236, 344], [236, 354], [260, 354], [269, 352], [292, 354], [302, 353], [323, 355], [326, 357], [335, 358], [336, 354], [347, 354], [350, 351], [355, 351], [357, 348], [359, 348], [359, 338], [351, 340], [349, 337], [352, 332], [357, 332], [357, 329], [359, 328], [358, 320], [361, 319], [361, 314], [359, 313], [359, 304], [357, 303], [357, 294], [348, 294], [347, 297], [342, 297], [340, 303], [337, 294], [288, 294], [245, 289], [232, 289], [231, 293], [230, 289], [222, 289], [222, 297], [218, 298], [218, 302], [216, 303], [216, 308], [218, 308], [220, 312], [218, 313], [216, 311]],
    To2d2(Gen3d_01()),
    To2d2(Gen3d_02()),
    To2d2(Gen3d_11()),
    To2d2(Gen3d_12()),
    To2d2(Gen3d_13()),
    ]

  polygon_3d= np.random.choice(polygons)
  if np.array(polygon_3d).shape[1]==2:
    polygon_2d= polygon_3d
    polygon_3d= np.array(polygon_3d)
    polygon_3d= np.hstack((polygon_3d,np.zeros((polygon_3d.shape[0],1))))
  else:
    polygon_2d= To2d(polygon_3d)

  t0= time.time()
  center3d= PolygonCentroid3D(polygon_3d)
  t3d= (time.time()-t0)*1e3

  t0= time.time()
  center2d= PolygonCentroid2D(polygon_2d)
  t2d= (time.time()-t0)*1e3

  print('# of polygon points: {}'.format(len(polygon_3d)))
  print('Center: {}: {}'.format('PolygonCentroid3D', center3d))
  print('Center: {}: {}'.format('PolygonCentroid2D', center2d))
  print('Diff: {}'.format(np.abs(center3d[:2]-center2d)))
  print('Time: {}: {} ms'.format('PolygonCentroid3D', t3d))
  print('Time: {}: {} ms'.format('PolygonCentroid2D', t2d))

