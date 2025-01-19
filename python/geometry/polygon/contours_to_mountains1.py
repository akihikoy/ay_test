#!/usr/bin/python3
#\file    contours_to_mountains1.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jun.30, 2020
import numpy as np
#from collections import namedtuple
from polygon_overlap import PolygonOverlap


class TContour(object):
  def __init__(self, level=None, points=None, outer=None, inner=None, idx=None):
    self.level  = level
    self.points = points
    self.outer  = outer
    self.inner  = inner if inner is not None else []
    self.idx    = idx

'''
Convert multi-level contours obtained by FindMultilevelContours to a better-organized form.
  Input: Multi-level contours obtained by FindMultilevelContours.
    contours: Multi-level contours.
    contours= <list>[contours[i]|i=0,1,...]
      contours[i]= <tuple>(level, subcontours)
        level: Level (height) at i.
        subcontours: Contours at level.
        subcontours= <list>[subcontours[j]|j=0,1,...]
          subcontours[j]: A contour; set of contour points.
          subcontours[j]= <np.array>[[[x0,y0]], [[x1,y1]], ...]
  Output: Multi-level contours with TContour (NOTE: original data memory is reused).
    contours: Multi-level contours.
    contours= <list>[contours[i]|i=0,1,...]
      contours[i]: Contours at a level.
      contours[i]= <list>[subcontours[j]|j=0,1,...]
        subcontours[j]: A contour.
        subcontours[j]= TContour.
          subcontours[j].level: Level (height), which should be common for all j in the same i.
          subcontours[j].points= <np.array>[[x0,y0], [x1,y1], ...]
          subcontours[j].outer: Outer contour.
          subcontours[j].inner: Inner contours (a list of TContour).
          subcontours[j].idx= (i,j)
'''
def LoadFromCVMultilevelContours(cv_contours):
  return [[TContour(level,subcontour.reshape(len(subcontour),2),idx=(i,j)) for j,subcontour in enumerate(subcontours)] for i,(level,subcontours) in enumerate(cv_contours)]

#Filter contours:
#  Removing small subcontours whose bounding box size is smaller than [lx,ly].
#  Optionally, removing near-bound points.
#  Optionally, removing corner points.
def FilterContours(contours, lx, ly, remove_boundary_points=False, remove_corner_points=False):
  xmax,ymax= np.max(sum([[np.max(subcontour.points,0) for subcontour in subcontours] for subcontours in contours],[]),0)
  xmin,ymin= np.min(sum([[np.min(subcontour.points,0) for subcontour in subcontours] for subcontours in contours],[]),0)
  new_contours= []
  for subcontours in contours:
    new_subcontours= []
    for subcontour in subcontours:
      size= np.max(subcontour.points,0)-np.min(subcontour.points,0)
      if size[0]>=lx and size[1]>=ly:
        new_subcontour= subcontour
        if remove_boundary_points:
          new_subcontour.points= [[x,y] for x,y in new_subcontour.points if all((x>xmin,x<xmax,y>ymin,y<ymax))]
        if remove_corner_points:
          new_subcontour.points= [[x,y] for x,y in new_subcontour.points if ((x>xmin and x<xmax) or (y>ymin and y<ymax))]
        if len(new_subcontour.points)>0:
          size= np.max(new_subcontour.points,0)-np.min(new_subcontour.points,0)
          if size[0]>=lx and size[1]>=ly:
            new_subcontours.append(new_subcontour)
    if len(new_subcontours)>0:
      new_contours.append(new_subcontours)
  for i,subcontours in enumerate(new_contours):
    for j,subcontour in enumerate(subcontours):
      subcontour.idx= (i,j)
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
      #if the inner bounding box is included in the outer bounding box
      if all((bounds_outer[0][0]<bounds[0][0],bounds_outer[0][1]<bounds[0][1], bounds[1][0]<bounds_outer[1][0],bounds[1][1]<bounds_outer[1][1])):
        return outer_contour
  if method=='bb_overlap':
    bounds= (np.min(contour.points,0),np.max(contour.points,0))
    for outer_contour in outer_contours:
      bounds_outer= (np.min(outer_contour.points,0),np.max(outer_contour.points,0))
      #if the inner bounding box is overlapping in the outer bounding box
      xcond= ((bounds_outer[0][0]<=bounds[0][0] and bounds[0][0]<=bounds_outer[1][0]) or (bounds_outer[0][0]<=bounds[1][0] and bounds[1][0]<=bounds_outer[1][0]))
      ycond= ((bounds_outer[0][1]<=bounds[0][1] and bounds[0][1]<=bounds_outer[1][1]) or (bounds_outer[0][1]<=bounds[1][1] and bounds[1][1]<=bounds_outer[1][1]))
      if xcond and ycond:
        return outer_contour
  if method=='polygon_overlap':
    bounds= (np.min(contour.points,0),np.max(contour.points,0))
    for outer_contour in outer_contours:
      bounds_outer= (np.min(outer_contour.points,0),np.max(outer_contour.points,0))
      #if the inner bounding box does not overlap with the outer bounding box
      if any((bounds_outer[1][0]<bounds[0][0], bounds[1][0]<bounds_outer[0][0], bounds_outer[1][1]<bounds[0][1], bounds[1][1]<bounds_outer[0][1])):
        continue
      if PolygonOverlap(contour.points,outer_contour.points):
        return outer_contour

#Analyzing a hierarchical structure of contours.
def AnalyzeContoursHierarchy(contours, method_findouter='polygon_overlap'):
  for k,subcontours in enumerate(contours[:-1]):
    for subcontour in subcontours:
      #Find outer contour of subcontour in contours[k+1] that includes subcontour.
      subcontour_outer= FindOuterContour(subcontour, contours[k+1], method_findouter)
      if subcontour_outer is not None:
        subcontour.outer= subcontour_outer
        subcontour_outer.inner.append(subcontour)

#Get a list of vertex subcontours from contours by using its hierarchy.
def GetVertexContours(contours):
  return sum(([subcontour for subcontour in subcontours if len(subcontour.inner)==0] for subcontours in contours), [])

##Get a list of valley subcontours from contours by using its hierarchy.
#def GetValleyContours(contours):
  #return sum(([subcontour for subcontour in subcontours if len(subcontour.inner)>1] for subcontours in contours), [])

#Get a list of valley subcontours from contours by using its hierarchy
#  where the valley level is larger than min_level.
def GetValleyContours(contours, min_level):
  def max_level(subcontour, base_level):
    if len(subcontour.inner)==0:
      return abs(subcontour.level-base_level)
    max_lv= 0
    for subcontour_inner in subcontour.inner:
      level= max_level(subcontour_inner, base_level)
      if max_lv<level:  max_lv= level
    return max_lv
  return sum(([subcontour for subcontour in subcontours if len(subcontour.inner)>1 and max_level(subcontour,subcontour.level)>min_level]
              for subcontours in contours), [])

#Write structure of contours into a file.
def WriteContoursStructure(file_name, contours, vertices=None, valleys=None):
  with open(file_name,'w') as fp:
    fp.write('#contours= \n')
    for subcontours in contours:
      fp.write('lv{0} {1}cntrs:'.format(subcontours[0].level, len(subcontours)) )
      for subcontour in subcontours:
        fp.write(' <@{idx} {l}pts {size[0]}x{size[1]}>'.format(idx=subcontour.idx, l=len(subcontour.points), size=np.max(subcontour.points,0)-np.min(subcontour.points,0)) )
      fp.write('\n')
    subcontours_to_str= lambda subcontours: ' '.join(['@{0}'.format(c.idx) for c in subcontours])
    if vertices:
      fp.write('#vertices= {0}\n'.format(subcontours_to_str(vertices)) )
      fp.write('#mountains= \n')
      for k,v in enumerate(vertices):
        outers= []
        vv= v
        while vv.outer is not None:  outers.append(vv.outer); vv= vv.outer
        fp.write('m{0} lv{1} @{2} [{3}] {4}\n'.format(k, v.level, v.idx, subcontours_to_str(v.inner), subcontours_to_str(outers) ))
    if valleys:
      fp.write('#valleys= \n')
      for k,v in enumerate(valleys):
        fp.write('v{0} lv{1} @{2} [{3}]\n'.format(k, v.level, v.idx, subcontours_to_str(v.inner) ))

def WriteContour(fp, subcontour):
  x,y= subcontour.points[-1]; fp.write('{0} {1} {2}\n'.format(x,y,subcontour.level))
  for x,y in subcontour.points:
    fp.write('{0} {1} {2}\n'.format(x,y,subcontour.level))

#Write contours into a file for plot contours per level.
#  Each "dataset" is separated by 2 blank lines, includes contours at the same level, each of which is separated by single blank line.
def WriteMultilevelContours(file_name, contours):
  with open(file_name,'w') as fp:
    for subcontours in contours:
      fp.write('#lv{0}\n'.format(subcontours[0].level))
      for subcontour in subcontours:
        fp.write('#@{0}\n'.format(subcontour.idx))
        WriteContour(fp, subcontour)
        fp.write('\n')
      fp.write('\n')

#Write mountain contours into a file for plot contours per mountain.
#  Each "dataset" is separated by 2 blank lines, includes contours in the same mountain, each of which is separated by single blank line.
def WriteMountainContours(file_name, vertices):
  with open(file_name,'w') as fp:
    for k,subcontour in enumerate(vertices):
      fp.write('#m{0} lv{1}\n'.format(k, subcontour.level))
      while True:
        fp.write('#@{0}\n'.format(subcontour.idx))
        WriteContour(fp, subcontour)
        fp.write('\n')
        if subcontour.outer is None:  break
        subcontour= subcontour.outer
      fp.write('\n')


if __name__=='__main__':
  import pickle
  cv_contours= pickle.load(open('data/mlcontours1.dat','rb'), encoding='latin1')
  #cv_contours= pickle.load(open('data/mlcontours2.dat','rb'), encoding='latin1')
  #cv_contours= pickle.load(open('data/mlcontours2a.dat','rb'), encoding='latin1')
  contours= LoadFromCVMultilevelContours(cv_contours)

  #Print structure of contours:
  print('# Original:')
  WriteContoursStructure('/dev/stdout', contours)
  #Write for plot:
  WriteMultilevelContours('/tmp/contours0.dat', contours)

  #Filtering contours:
  contours= FilterContours(contours, 60, 60)
  print('# Filtered:')
  WriteContoursStructure('/dev/stdout', contours)
  WriteMultilevelContours('/tmp/contours1.dat', contours)
  print('# Contours: Plot by:')
  level_min= min(contours[0][0].level,contours[-1][0].level)
  level_max= max(contours[0][0].level,contours[-1][0].level)
  ps_per_lv= 3.9/(level_max-level_min)
  print('# Original and filtered contours:')
  print('''qplot -x /tmp/contours0.dat w l /tmp/contours1.dat w l lw 2 ''')
  print('# Filtered contours with variable-size points (ps=level):')
  print('''qplot -x /tmp/contours1.dat u 1:2:'(0.1+{ps_per_lv}*($3-{level_min}))' w p pt 6 ps variable '''.format(ps_per_lv=ps_per_lv,level_min=level_min))
  print('# Filtered contours with lines (line color per level):')
  print('''bash -c 'p=""; for i in `seq 0 {nc}`;do p="$p /tmp/contours1.dat index $i w l";done; qplot -x $p' '''.format(nc=len(contours)-1))
  print('# Filtered contours with variable-size points (ps=level; point color per level):')
  print('''bash -c 'p=""; for i in `seq {nc} -1 0`;do p="$p /tmp/contours1.dat index $i u 1:2:(0.1+{ps_per_lv}*(\$3-{level_min})) w p pt 6 ps variable";done; qplot -x $p' '''.format(nc=len(contours)-1,ps_per_lv=ps_per_lv,level_min=level_min))

  #Analyzing the hierarchical structure.
  AnalyzeContoursHierarchy(contours)
  vertices= GetVertexContours(contours)
  WriteContoursStructure('/dev/stdout', contours, vertices)

  #Write hierarchical structure.
  WriteMountainContours('/tmp/m_contours.dat', vertices)
  print('# Mountain contours: Plot by:')
  level_min= min(contours[0][0].level,contours[-1][0].level)
  level_max= max(contours[0][0].level,contours[-1][0].level)
  ps_per_lv= 3.9/(level_max-level_min)
  print('# 0th Mountain contours with variable-size points (ps=level):')
  print('''qplot -x /tmp/m_contours.dat u 1:2:'(0.1+{ps_per_lv}*($3-{level_min}))' index 0 w p pt 6 ps variable '''.format(ps_per_lv=ps_per_lv,level_min=level_min))
  print('# Mountain contours with lines (line color per mountain):')
  print('''bash -c 'p=""; for i in `seq 0 {nv}`;do p="$p /tmp/m_contours.dat index $i w l";done; qplot -x $p' '''.format(nv=len(vertices)-1))
  print('# Mountain contours with variable-size points (ps=level; point color per mountain):')
  print('''bash -c 'p=""; for i in `seq 0 {nv}`;do p="$p /tmp/m_contours.dat u 1:2:(0.1+{ps_per_lv}*(\$3-{level_min})) index $i w p pt 6 ps variable";done; qplot -x $p' '''.format(nv=len(vertices)-1,ps_per_lv=ps_per_lv,level_min=level_min))
  print('# i-th mountain contours (line color per level):')
  print('''bash -c 'i=0; p=""; for e in `seq 0 60`;do p="$p /tmp/m_contours.dat index $i ev :::$e::$e w l";done; qplot -x $p' ''')

