#!/usr/bin/python3
#\file    contours_to_mountains2.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jul.02, 2020
from contours_to_mountains1 import *
from weighted_ellipse_fit2 import SqErrorFromEllipse,SampleWeightedEllipseFit2D
from ellipse_point_in_out import PointInEllipse

#Approximate each contour in a mountain by an ellipse.
#  vertex: Vertex contour of the mountain.
#  w_param: Parameter to convert a distance from ellipse to weight.
#  min_ellipses: Min number of ellipses per mountain. 0 for no limit.
#  max_ellipses: Max number of ellipses per mountain. 0 for no limit.
#  max_area_increase: If the area of the current ellipse is greater than max_area_increase-times larger
#    than that of the inner ellipse, it stops adding ellipses.
def ApproxMountainByEllipses(vertices, w_param=0.01, min_ellipses=3, max_ellipses=10, max_area_increase=3.0):
  mountains= []
  for subcontour in vertices:
    ellipses= []
    W= [1.0]*len(subcontour.points)
    c,r1,r2,angle= SampleWeightedEllipseFit2D(subcontour.points, W)
    ellipses.append((subcontour,c,r1,r2,angle))
    while subcontour.outer is not None:
      if max_ellipses>0 and len(ellipses)>=max_ellipses:  break
      subcontour= subcontour.outer
      #W= [1.0/(1.0+w_param*SqErrorFromEllipse([x,y],c,r1,r2,angle)) for x,y in subcontour.points]
      W= [1.0/(1.0+w_param*np.sqrt(SqErrorFromEllipse([x,y],c,r1,r2,angle))) for x,y in subcontour.points]
      #W= [1.0 if np.sqrt(SqErrorFromEllipse([x,y],c,r1,r2,angle))<w_param else 0.0 for x,y in subcontour.points]
      #print 'W',W
      c,r1,r2,angle= SampleWeightedEllipseFit2D(subcontour.points, W, centroid=c)
      #If the ellipse is smaller than the previous one, stop the process.
      #Check the size difference with the product of two axis lengths:
      #if r1*r2 < ellipses[-1][2]*ellipses[-1][3]:  break
      #Check the size difference per axis length:
      if len(ellipses)>=min_ellipses and r1<ellipses[-1][2] or r2<ellipses[-1][3]:  break
      #Check the size increase.
      if len(ellipses)>=min_ellipses and r1*r2>max_area_increase*ellipses[-1][2]*ellipses[-1][3]:  break
      ellipses.append((subcontour,c,r1,r2,angle))
    mountains.append(ellipses)
  return mountains

def WriteMountainEllipses(file_name, mountains):
  with open(file_name,'w') as fp:
    for ellipses in mountains:
      for subcontour,c,r1,r2,angle in ellipses:
        for th in np.linspace(0, 2*np.pi, 1000):
          x= c[0] + r1*np.cos(angle)*np.cos(th) - r2*np.sin(angle)*np.sin(th)
          y= c[1] + r1*np.sin(angle)*np.cos(th) + r2*np.cos(angle)*np.sin(th)
          fp.write('{0} {1} {2}\n'.format(x,y,subcontour.level))
        fp.write('\n')
      fp.write('\n')

##Get a level at point on a mountain (==ellipses).
##Return None if point is not on the mountain
#def PointLevelOnMountain(ellipses, point):
  #for subcontour,c,r1,r2,angle in ellipses:
    #if PointInEllipse(point, c,r1,r2,angle):  return subcontour.level
  #return None

##Get a level at point on mountains.
#def PointLevelOnMountains(mountains, point):
  #levels= []
  #for ellipses in mountains:
    #level= PointLevelOnMountain(ellipses, point)
    #if level is not None:  levels.append(level)
  #if len(levels)>0:
    #return min(levels)
  #return None

##Convert mountains to a depth image.
#def MountainsToDepthImg(mountains, width, height):
  #none_filter= lambda lv: 255 if lv is None else lv
  #return np.array([[none_filter(PointLevelOnMountains(mountains, [x,y])) for x in range(width)]
                   #for y in range(height)],dtype=np.uint8)

import cv2
#Convert mountains to a depth image.
def MountainsToDepthImg(mountains, width, height):
  img= np.ones([height,width],dtype=np.uint16)*np.iinfo(np.uint16).max
  ellipses= sum(mountains,[])
  ellipses.sort(key=lambda e:e[0].level)
  for subcontour,c,r1,r2,angle in reversed(ellipses):
    poly= [[c[0] + r1*np.cos(angle)*np.cos(th) - r2*np.sin(angle)*np.sin(th),
            c[1] + r1*np.sin(angle)*np.cos(th) + r2*np.cos(angle)*np.sin(th)
            ] for th in np.linspace(0, 2*np.pi, 200)]
    col= int(subcontour.level)
    cv2.fillPoly(img, [np.array(poly,np.int32).reshape((-1,1,2))], col)
  return img


if __name__=='__main__':
  import pickle
  cv_contours= pickle.load(open('data/mlcontours1.dat','rb'), encoding='latin1')
  #cv_contours= pickle.load(open('data/mlcontours2.dat','rb'), encoding='latin1')
  #cv_contours= pickle.load(open('data/mlcontours2a.dat','rb'), encoding='latin1')
  contours= LoadFromCVMultilevelContours(cv_contours)

  #Filtering contours:
  print('# Filtering contours...')
  contours= FilterContours(contours, 60, 60)
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
  print('# Analyzing the hierarchical structure...')
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

  #Approximate each contour in a mountain by an ellipse.
  print('# Approximate each contour in a mountain by an ellipse:')
  mountains= ApproxMountainByEllipses(vertices)
  WriteMountainEllipses('/tmp/m_ellipses.dat', mountains)
  print('# Ellipses: Plot by:')
  level_min= min(contours[0][0].level,contours[-1][0].level)
  level_max= max(contours[0][0].level,contours[-1][0].level)
  ps_per_lv= 3.9/(level_max-level_min)
  print('# Ellipses of mountain with lines (line color per mountain):')
  print('''bash -c 'p=""; for i in `seq 0 {nv}`;do p="$p /tmp/m_ellipses.dat index $i w l";done; qplot -x $p' '''.format(nv=len(vertices)-1))
  print('# First 10 ellipses of mountain with lines (line color per mountain):')
  print('''bash -c 'p=""; for i in `seq 0 {nv}`;do p="$p /tmp/m_ellipses.dat index $i ev :::0::10 w l";done; qplot -x $p' '''.format(nv=len(vertices)-1))
  print('# Ellipses of mountain with lines with data points (line color per mountain):')
  print('''bash -c 'p=""; for i in `seq 0 {nv}`;do p="$p /tmp/m_contours.dat index $i w dots lt $((i+1)) /tmp/m_ellipses.dat index $i w l lt $((i+1))";done; qplot -x $p' '''.format(nv=len(vertices)-1))
  print('# First 10 ellipses of mountain with lines with data points (line color per mountain):')
  print('''bash -c 'p=""; for i in `seq 0 {nv}`;do p="$p /tmp/m_contours.dat index $i w dots lt $((i+1)) /tmp/m_ellipses.dat index $i ev :::0::10 w l lt $((i+1))";done; qplot -x $p' '''.format(nv=len(vertices)-1))
  print('# Ellipses of mountain with variable-size points (ps=level; point color per mountain):')
  print('''bash -c 'p=""; for i in `seq 0 {nv}`;do p="$p /tmp/m_ellipses.dat u 1:2:(0.1+{ps_per_lv}*(\$3-{level_min})) index $i w p pt 6 ps variable";done; qplot -x $p' '''.format(nv=len(vertices)-1,ps_per_lv=ps_per_lv,level_min=level_min))
  print('# i-th ellipses of mountain (line color per level):')
  print('''bash -c 'i=0; p=""; for e in `seq 0 60`;do p="$p /tmp/m_ellipses.dat index $i ev :::$e::$e w l";done; qplot -x $p' ''')
  print('# i-th ellipses of mountain with data points (line color per level):')
  print('''bash -c 'i=0; p=""; for e in `seq 0 10`;do p="$p /tmp/m_contours.dat index $i ev :::$e::$e w p lt $((e+1)) /tmp/m_ellipses.dat index $i ev :::$e::$e w l lt $((e+1))";done; qplot -x $p' ''')

  #Convert mountains to a depth image.
  print('# Convert mountains to a depth image.')
  width,height= np.max(sum([[np.max(subcontour.points,0) for subcontour in subcontours] for subcontours in contours],[]),0)
  depth_img= MountainsToDepthImg(mountains, width, height)
  depth_img= cv2.cvtColor(depth_img.astype('uint8'), cv2.COLOR_GRAY2BGR)
  import cv2
  cv2.imwrite('/tmp/depth_img.png', depth_img)
  print('''display /tmp/depth_img.png''')
  cv2.imshow('depth', depth_img)
  while cv2.waitKey() not in list(map(ord,[' ','q'])):  pass

