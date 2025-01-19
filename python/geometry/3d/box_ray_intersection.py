#!/usr/bin/python3
#\file    box_ray_intersection.py
#\brief   Get an intersection between a ray and a cube.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Sep.09, 2019

'''
Based on the Ray-box intersection algorithm described in:
    Amy Williams, Steve Barrus, R. Keith Morley, and Peter Shirley
    "An Efficient and Robust Ray-Box Intersection Algorithm"
    Journal of graphics tools, 10(1):49-54, 2005

ray_o: Origin of the ray (x,y,z)
ray_d: Direction of the ray (x,y,z)
box_min: Min position of the box (x,y,z)
box_max: Max position of the box (x,y,z)
'''
def BoxRayIntersection(ray_o, ray_d, box_min, box_max):
  sign= lambda x: 1 if x<0 else 0
  INF= 1.0e100
  EPS= 1.0e-100
  ray_id= [(1.0/d if abs(d)>EPS else (INF if d>=0.0 else -INF)) for d in ray_d]
  ray_sign= [sign(id) for id in ray_id]
  box= [box_min, box_max]

  tmin= (box[ray_sign[0]][0] - ray_o[0]) * ray_id[0]
  tmax= (box[1-ray_sign[0]][0] - ray_o[0]) * ray_id[0]
  tymin= (box[ray_sign[1]][1] - ray_o[1]) * ray_id[1]
  tymax= (box[1-ray_sign[1]][1] - ray_o[1]) * ray_id[1]
  if (tmin > tymax) or (tymin > tmax):
    return None,None
  if tymin > tmin:
    tmin= tymin
  if tymax < tmax:
    tmax= tymax
  tzmin= (box[ray_sign[2]][2] - ray_o[2]) * ray_id[2]
  tzmax= (box[1-ray_sign[2]][2] - ray_o[2]) * ray_id[2]
  if (tmin > tzmax) or (tzmin > tmax):
    return None,None
  if tzmin > tmin:
    tmin= tzmin;
  if tzmax < tmax:
    tmax= tzmax;
  return tmin, tmax


if __name__=='__main__':
  import numpy as np
  from random import random
  import matplotlib.pyplot as pyplot
  from plot_cube import PlotCube

  #box_min= [-1., -1., -0.5]
  #box_max= [1., 1., 0.5]
  #ray_o= np.array([1.1, 0.9, 0.])
  #ray_d= np.array([1., 0, 0])
  rnd= lambda: 2.0*random()-1.0
  box_min= [-random(), -random(), -random()]
  box_max= [ random(),  random(),  random()]
  ray_o= np.array([rnd(), rnd(), rnd()])
  ray_d= np.array([rnd(), rnd(), rnd()])

  print('box_min',box_min)
  print('box_max',box_max)
  print('ray_o',ray_o)
  print('ray_d',ray_d)

  tmin,tmax= BoxRayIntersection(ray_o, ray_d, box_min, box_max)
  print('tmin,tmax', tmin,tmax)
  if tmin is not None and tmax is not None:
    pmin= np.array(ray_o)+tmin*ray_d
    pmax= np.array(ray_o)+tmax*ray_d

  cube_points= np.array( [[box_min[0], box_min[1], box_min[2]],
                          [box_max[0], box_min[1], box_min[2]],
                          [box_max[0], box_max[1], box_min[2]],
                          [box_min[0], box_max[1], box_min[2]],
                          [box_min[0], box_min[1], box_max[2]],
                          [box_max[0], box_min[1], box_max[2]],
                          [box_max[0], box_max[1], box_max[2]],
                          [box_min[0], box_max[1], box_max[2]]])
  fig= pyplot.figure()
  ax= fig.add_subplot(111, projection='3d')
  PlotCube(ax, cube_points)
  ax.plot([ray_o[0]], [ray_o[1]], [ray_o[2]],'o')
  ax.quiver([ray_o[0]], [ray_o[1]], [ray_o[2]], [ray_d[0]], [ray_d[1]], [ray_d[2]], pivot='tail')
  if tmin is not None and tmax is not None:
    ax.plot([pmin[0]], [pmin[1]], [pmin[2]],'<')
    ax.plot([pmax[0]], [pmax[1]], [pmax[2]],'>')

  ax.set_xlim([-1.5,1.5])
  ax.set_ylim([-1.5,1.5])
  ax.set_zlim([-1.5,1.5])
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  pyplot.show()

