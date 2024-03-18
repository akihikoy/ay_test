#!/usr/bin/python
#\file    polygon_center_comp.py
#\brief   Comparison of polygon center estimation.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Mar.18, 2024
#from polygon_com import PolygonCentroid
#from polygon_angle2 import PolygonCentroid
from polygon_com_2d import PolygonCentroid2D, PolygonCentroid3D
from polygon_min_area_rect import MinAreaRect



def PolygonCenter_Mean(polygon):
  center= np.mean(polygon,axis=0)
  return center

def PolygonCenter_Median(polygon):
  center= np.median(polygon,axis=0)
  return center

def PolygonCenter_MinAreaRect(polygon):
  bb_center,bb_size,bb_angle= MinAreaRect(np.array(polygon)[:,:2])
  #center= np.array([bb_center[0], bb_center[1], np.mean(np.array(polygon)[:,2])])
  center= np.array([bb_center[0], bb_center[1]])
  return center

def PolygonCenter_PCA(polygon):
  polygon= np.array(polygon)
  center= PolygonCentroid3D(np.hstack((polygon,np.zeros((polygon.shape[0],1)))))
  return center[:2]

def PolygonCenter_CoM(polygon):
  center= PolygonCentroid2D(polygon)
  return center

if __name__=='__main__':
  from gen_data import *
  import matplotlib.pyplot as plt
  import time

  polygons=[
    [[469, 232], [464, 232], [461, 230], [461, 222], [458, 222], [457, 219], [451, 219], [450, 221], [448, 221], [448, 226], [444, 228], [441, 224], [441, 219], [422, 217], [421, 211], [423, 209], [424, 200], [423, 190], [417, 190], [416, 198], [413, 200], [412, 214], [406, 218], [390, 216], [387, 214], [364, 213], [358, 210], [346, 211], [345, 208], [320, 207], [320, 217], [316, 221], [316, 225], [318, 226], [318, 230], [322, 232], [316, 234], [316, 242], [318, 243], [318, 245], [316, 247], [316, 252], [312, 253], [312, 258], [316, 259], [316, 267], [320, 270], [320, 273], [328, 274], [329, 271], [333, 271], [334, 275], [377, 275], [394, 279], [401, 278], [414, 281], [429, 281], [429, 279], [433, 276], [434, 281], [451, 283], [452, 281], [457, 281], [459, 279], [459, 275], [463, 273], [463, 266], [459, 266], [456, 269], [453, 264], [457, 263], [458, 261], [459, 263], [463, 263], [463, 254], [465, 252], [465, 248], [467, 247], [467, 242], [465, 241], [469, 240]],
    [[212, 311], [212, 315], [214, 317], [214, 326], [216, 327], [212, 329], [212, 337], [214, 338], [216, 350], [219, 351], [233, 351], [234, 342], [236, 344], [236, 354], [260, 354], [269, 352], [292, 354], [302, 353], [323, 355], [326, 357], [335, 358], [336, 354], [347, 354], [350, 351], [355, 351], [357, 348], [359, 348], [359, 338], [351, 340], [349, 337], [352, 332], [357, 332], [357, 329], [359, 328], [358, 320], [361, 319], [361, 314], [359, 313], [359, 304], [357, 303], [357, 294], [348, 294], [347, 297], [342, 297], [340, 303], [337, 294], [288, 294], [245, 289], [232, 289], [231, 293], [230, 289], [222, 289], [222, 297], [218, 298], [218, 302], [216, 303], [216, 308], [218, 308], [220, 312], [218, 313], [216, 311]],
    To2d(Gen3d_01()),
    To2d(Gen3d_02()),
    To2d(Gen3d_11()),
    To2d(Gen3d_12()),
    To2d(Gen3d_13()),
    ]

  polygon= np.random.choice(polygons)

  fig= plt.figure()
  markers= ['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_']
  ax= fig.add_subplot(1,1,1)
  def plotpoly(poly, **kwargs):
    poly= list(poly)
    poly= poly+[poly[0]]
    return ax.plot(np.array(poly)[:,0], np.array(poly)[:,1], **kwargs)
  plotpoly(polygon, color='blue',  label='polygon')
  for method in ('Mean','Median','MinAreaRect','PCA','CoM'):
    t0= time.time()
    center= eval('PolygonCenter_{}(polygon)'.format(method))
    print 'Time: {}: {} ms'.format(method,(time.time()-t0)*1e3)
    ax.scatter(*center, label=method, marker=markers[sum(ord(s) for s in method)%len(markers)], s=64)
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.grid(True)
  ax.legend(loc='lower left')
  plt.show()
