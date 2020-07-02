#!/usr/bin/python
#\file    contours_to_mountains1.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jun.30, 2020
import numpy as np
from collections import namedtuple
from polygon_overlap import PolygonOverlap


class TContour(object):
  def __init__(self, level=None, points=None, outer=None, inner=None, idx=None):
    self.level  = level
    self.points = points
    self.outer  = outer
    self.inner  = inner if inner is not None else []
    self.idx    = idx

'''
Input: Multi-level contours obtained by FindMultilevelContours.
Output: Reshaped multi-level contours (NOTE: contours is overwritten):
  contours: Multi-level contours.
  contours= <list>[contours[i]|i=0,1,...]
    contours[i]= <tuple>(level, subcontours)
      level: Level (height) at i.
      subcontours: Contours at level.
      subcontours= <list>[subcontours[j]|j=0,1,...]
        subcontours[j]: A contour; set of contour points.
        subcontours[j]= <np.array>[[x0,y0], [x1,y1], ...]
'''
def ReshapeMultilevelContours(contours):
  for level, subcontours in contours:
    for subcontour in subcontours:
      subcontour.shape= (len(subcontour),2)

#Filter contours:
#  Removing small subcontours whose bounding box size is smaller than [lx,ly].
#  Optionally, removing near-bound points.
#  Optionally, removing corner points.
def FilterContours(contours, lx, ly, remove_boundary_points=False, remove_corner_points=False):
  xmax,ymax= np.max(sum([[np.max(subcontour,0) for subcontour in subcontours] for level,subcontours in contours],[]),0)
  xmin,ymin= np.min(sum([[np.min(subcontour,0) for subcontour in subcontours] for level,subcontours in contours],[]),0)
  new_contours= []
  for level, subcontours in contours:
    new_subcontours= []
    for subcontour in subcontours:
      size= np.max(subcontour,0)-np.min(subcontour,0)
      if size[0]>=lx and size[1]>=ly:
        new_subcontour= subcontour
        if remove_boundary_points:
          new_subcontour= [[x,y] for x,y in new_subcontour if all((x>xmin,x<xmax,y>ymin,y<ymax))]
        if remove_corner_points:
          new_subcontour= [[x,y] for x,y in new_subcontour if ((x>xmin and x<xmax) or (y>ymin and y<ymax))]
        if len(new_subcontour)>0:
          size= np.max(new_subcontour,0)-np.min(new_subcontour,0)
          if size[0]>=lx and size[1]>=ly:
            new_subcontours.append(new_subcontour)
    if len(new_subcontours)>0:
      new_contours.append((level, new_subcontours))
  return new_contours

'''
Find an outer contour that includes the given contour from outer_contours.
  method=='bb_inclusion':  #Fast, innaccurate.
  method=='bb_overlap':  #Fast, innaccurate, more candidates than bb_inclusion.
  method=='polygon_overlap':  #Slow, more accurate.
'''
def FindOuterContour(contour, outer_contours, method):
  if method=='bb_inclusion':
    bounds= (np.min(contour.points,0),np.max(contour.points,0))
    for outer_contour in outer_contours:
      bounds_outer= (np.min(outer_contour.points,0),np.max(outer_contour.points,0))
      #if an inner bounding box is included in an outer bounding box
      if all((bounds_outer[0][0]<bounds[0][0],bounds_outer[0][1]<bounds[0][1], bounds[1][0]<bounds_outer[1][0],bounds[1][1]<bounds_outer[1][1])):
        return outer_contour
  if method=='bb_overlap':
    bounds= (np.min(contour.points,0),np.max(contour.points,0))
    for outer_contour in outer_contours:
      bounds_outer= (np.min(outer_contour.points,0),np.max(outer_contour.points,0))
      #if an inner bounding box is overlapping in an outer bounding box
      xcond= ((bounds_outer[0][0]<=bounds[0][0] and bounds[0][0]<=bounds_outer[1][0]) or (bounds_outer[0][0]<=bounds[1][0] and bounds[1][0]<=bounds_outer[1][0]))
      ycond= ((bounds_outer[0][1]<=bounds[0][1] and bounds[0][1]<=bounds_outer[1][1]) or (bounds_outer[0][1]<=bounds[1][1] and bounds[1][1]<=bounds_outer[1][1]))
      if xcond and ycond:
        return outer_contour
  if method=='polygon_overlap':
    for outer_contour in outer_contours:
      if PolygonOverlap(contour.points,outer_contour.points):
        return outer_contour

#Making a hierarchical structure of contours.
def GetHierarchicalContours(contours, method_findouter='polygon_overlap'):
  h_contours= [(level, [TContour(level,subcontour,idx=(k,i)) for i,subcontour in enumerate(subcontours)]) for k,(level,subcontours) in enumerate(contours)]
  for k,(level,subcontours) in enumerate(h_contours[:-1]):
    for subcontour in subcontours:
      #Find outer contour of subcontour in h_contours[k+1][1] that includes subcontour.
      subcontour_outer= FindOuterContour(subcontour, h_contours[k+1][1], method_findouter)
      if subcontour_outer is not None:
        subcontour.outer= subcontour_outer
        subcontour_outer.inner.append(subcontour)
  return h_contours

#Get a list of vertex subcontours from hierarchical contours.
def GetVertices(h_contours):
  vertices= []
  for level, subcontours in h_contours:
    for subcontour in subcontours:
      if len(subcontour.inner)==0:
        vertices.append(subcontour)
  return vertices

#Write structure of contours into a file.
def WriteContoursStructure(file_name, contours):
  with open(file_name,'w') as fp:
    for level, subcontours in contours:
      fp.write('{0} {1}'.format(level, len(subcontours)) )
      for subcontour in subcontours:
        fp.write(' <{l} {size}>'.format(l=len(subcontour),size=np.max(subcontour,0)-np.min(subcontour,0)) )
      fp.write('\n')

#Write contours into a file.
def WriteMultilevelContours(file_name, contours):
  with open(file_name,'w') as fp:
    for level, subcontours in contours:
      for subcontour in subcontours:
        #x,y= subcontour[-1]; fp.write('{0} {1} {2}\n'.format(x,y,level))
        for x,y in subcontour:
          fp.write('{0} {1} {2}\n'.format(x,y,level))
        fp.write('\n')
      fp.write('\n')

if __name__=='__main__':
  import pickle
  contours= pickle.load(open('data/mlcontours1.dat','rb'))
  #contours= pickle.load(open('data/mlcontours2.dat','rb'))
  ReshapeMultilevelContours(contours)

  #Print structure of contours:
  print '# Original:'
  WriteContoursStructure('/dev/stdout', contours)
  #Write for plot:
  WriteMultilevelContours('/tmp/contours0.dat', contours)

  #Filtering contours:
  contours= FilterContours(contours, 60, 60)
  print '# Filtered:'
  WriteContoursStructure('/dev/stdout', contours)
  WriteMultilevelContours('/tmp/contours1.dat', contours)
  print '# Contours: Plot by:'
  level_min= min(contours[0][0],contours[-1][0])
  level_max= max(contours[0][0],contours[-1][0])
  ps_per_lv= 3.9/(level_max-level_min)
  print '# Original and filtered contours:'
  print '''qplot -x /tmp/contours0.dat w l /tmp/contours1.dat w l lw 2 '''
  print '# Filtered contours with variable-size points (ps=level):'
  print '''qplot -x /tmp/contours1.dat u 1:2:'(0.1+{ps_per_lv}*($3-{level_min}))' w p pt 6 ps variable '''.format(ps_per_lv=ps_per_lv,level_min=level_min)
  print '# Filtered contours with lines (line color per level):'
  print '''bash -c 'p=""; for i in `seq 0 {nc}`;do p="$p /tmp/contours1.dat index $i w l";done; qplot -x $p' '''.format(nc=len(contours)-1)
  print '# Filtered contours with variable-size points (ps=level; point color per level):'
  print '''bash -c 'p=""; for i in `seq 0 {nc}`;do p="$p /tmp/contours1.dat index $i u 1:2:(0.1+{ps_per_lv}*(\$3-{level_min})) w p pt 6 ps variable";done; qplot -x $p' '''.format(nc=len(contours)-1,ps_per_lv=ps_per_lv,level_min=level_min)

  #Making a hierarchical structure.
  h_contours= GetHierarchicalContours(contours)
  vertices= GetVertices(h_contours)
  with open('/dev/stdout','w') as fp:
    fp.write('all contours= \n')
    for level, subcontours in h_contours:
      fp.write('{0} {1}'.format(level, len(subcontours)) )
      for subcontour in subcontours:
        fp.write(' {0}'.format(subcontour.idx) )
      fp.write('\n')
    fp.write('vertices= {0}\n'.format(' '.join(map(lambda v:str(v.idx),vertices))) )
    fp.write('vertices= \n')
    for v in vertices:
      outers= []
      vv= v
      while vv.outer is not None:  outers.append(vv.outer); vv= vv.outer
      fp.write('{0} {1} [{2}] {3}\n'.format(v.level, v.idx, ' '.join(map(lambda n:str(n.idx),v.inner)), ' '.join(map(lambda w:str(w.idx),outers)) ))

  #Write hierarchical structure.
  with open('/tmp/h_contours.dat','w') as fp:
    for subcontour in vertices:
      while True:
        fp.write('#{0}\n'.format(subcontour.idx))
        #x,y= subcontour.points[-1]; fp.write('{0} {1} {2}\n'.format(x,y,level))
        for x,y in subcontour.points:
          fp.write('{0} {1} {2}\n'.format(x,y,subcontour.level))
        fp.write('\n')
        if subcontour.outer is None:  break
        subcontour= subcontour.outer
      fp.write('\n')
  print '# Hierarchical contours: Plot by:'
  level_min= min(h_contours[0][0],h_contours[-1][0])
  level_max= max(h_contours[0][0],h_contours[-1][0])
  ps_per_lv= 3.9/(level_max-level_min)
  print '# Hierarchical contours with variable-size points (ps=level):'
  print '''qplot -x /tmp/h_contours.dat u 1:2:'(0.1+{ps_per_lv}*($3-{level_min}))' index 0 w p pt 6 ps variable '''.format(ps_per_lv=ps_per_lv,level_min=level_min)
  print '# Hierarchical contours with lines (line color per mountain):'
  print '''bash -c 'p=""; for i in `seq 0 {nv}`;do p="$p /tmp/h_contours.dat index $i w l";done; qplot -x $p' '''.format(nv=len(vertices)-1)
  print '# Hierarchical contours with variable-size points (ps=level; point color per mountain):'
  print '''bash -c 'p=""; for i in `seq 0 {nv}`;do p="$p /tmp/h_contours.dat u 1:2:(0.1+{ps_per_lv}*(\$3-{level_min})) index $i w p pt 6 ps variable";done; qplot -x $p' '''.format(nv=len(vertices)-1,ps_per_lv=ps_per_lv,level_min=level_min)
  print '# i-th mountain\'s contours (line color per level):'
  print '''bash -c 'i=0; p=""; for e in `seq 0 60`;do p="$p /tmp/h_contours.dat index $i ev :::$e::$e w l";done; qplot -x $p' '''

