#!/usr/bin/python
#\file    contours_to_mountains3.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jul.06, 2020
from contours_to_mountains2 import *


##Make a valley map where a valley pixel is filled by its level.
#def MakeValleyMap(valleys, width, height):
  #img= np.zeros([height,width],dtype=np.uint16)
  #for subcontour in reversed(valleys):
    #subimg= np.zeros([height,width],dtype=np.uint16)
    #col= subcontour.level
    #cv2.fillPoly(subimg, [subcontour.points.reshape((-1,1,2))], col)
    #for subcontour_inner in subcontour.inner:
      #cv2.fillPoly(subimg, [subcontour_inner.points.reshape((-1,1,2))], 0)
    #img[subimg>0]= subimg[subimg>0]
  #return img

##Make a valley map where a valley pixel is filled by its level.
#def MakeValleyMap(valleys, width, height):
  #img= np.zeros([height,width],dtype=np.uint16)
  #def draw_outers(subimg, subcontour):
    #if subcontour.outer is not None:
      #draw_outers(subimg, subcontour.outer)
    #col= subcontour.level
    #cv2.fillPoly(subimg, [subcontour.points.reshape((-1,1,2))], col)
  #for subcontour in reversed(valleys):
    #subimg= np.zeros([height,width],dtype=np.uint16)
    ##col= subcontour.level
    ##cv2.fillPoly(subimg, [subcontour.points.reshape((-1,1,2))], col)
    #draw_outers(subimg, subcontour)
    #for subcontour_inner in subcontour.inner:
      #cv2.fillPoly(subimg, [subcontour_inner.points.reshape((-1,1,2))], 0)
    #img[subimg>0]= subimg[subimg>0]
  #return img

#Make a valley map where a valley pixel is filled by its level.
def MakeValleyMap(valleys, width, height):
  valleys_and_outers= []
  for subcontour in reversed(valleys):
    while True:
      if id(subcontour) not in map(id,valleys_and_outers):
        valleys_and_outers.append(subcontour)
      if subcontour.outer is None:  break
      subcontour= subcontour.outer
  mountains= []  #mountain contours.
  for subcontour in reversed(valleys):
    for subcontour_inner in subcontour.inner:
      if id(subcontour_inner) not in map(id,mountains):
        mountains.append(subcontour_inner)
  subcontours= [(subcontour,'v') for subcontour in valleys_and_outers] \
              +[(subcontour,'m') for subcontour in mountains]
  subcontours.sort(key=lambda v:v[0].level)
  img= np.zeros([height,width],dtype=np.uint16)
  for subcontour,kind in reversed(subcontours):
    if kind=='v':
      col= subcontour.level
      cv2.fillPoly(img, [subcontour.points.reshape((-1,1,2))], col)
    else:
      cv2.fillPoly(img, [subcontour.points.reshape((-1,1,2))], 0)
  return img

if __name__=='__main__':
  import pickle
  #cv_contours= pickle.load(open('data/mlcontours1.dat','rb'))
  #cv_contours= pickle.load(open('data/mlcontours2.dat','rb'))
  #cv_contours= pickle.load(open('data/mlcontours2a.dat','rb'))
  cv_contours= pickle.load(open('data/mlcontours4a.dat','rb'))
  contours= LoadFromCVMultilevelContours(cv_contours)

  #Filtering contours:
  print '# Filtering contours...'
  contours= FilterContours(contours, 60, 60)
  WriteMultilevelContours('/tmp/contours1.dat', contours)
  print '# Contours: Plot by:'
  level_min= min(contours[0][0].level,contours[-1][0].level)
  level_max= max(contours[0][0].level,contours[-1][0].level)
  ps_per_lv= 3.9/(level_max-level_min)
  print '# Original and filtered contours:'
  print '''qplot -x /tmp/contours0.dat w l /tmp/contours1.dat w l lw 2 '''
  print '# Filtered contours with variable-size points (ps=level):'
  print '''qplot -x /tmp/contours1.dat u 1:2:'(0.1+{ps_per_lv}*($3-{level_min}))' w p pt 6 ps variable '''.format(ps_per_lv=ps_per_lv,level_min=level_min)
  print '# Filtered contours with lines (line color per level):'
  print '''bash -c 'p=""; for i in `seq 0 {nc}`;do p="$p /tmp/contours1.dat index $i w l";done; qplot -x $p' '''.format(nc=len(contours)-1)
  print '# Filtered contours with variable-size points (ps=level; point color per level):'
  print '''bash -c 'p=""; for i in `seq {nc} -1 0`;do p="$p /tmp/contours1.dat index $i u 1:2:(0.1+{ps_per_lv}*(\$3-{level_min})) w p pt 6 ps variable";done; qplot -x $p' '''.format(nc=len(contours)-1,ps_per_lv=ps_per_lv,level_min=level_min)

  #Analyzing the hierarchical structure.
  print '# Analyzing the hierarchical structure...'
  AnalyzeContoursHierarchy(contours)
  vertices= GetVertexContours(contours)
  valleys= GetValleyContours(contours, min_level=20)
  WriteContoursStructure('/dev/stdout', contours, vertices, valleys)

  #Write hierarchical structure.
  WriteMountainContours('/tmp/m_contours.dat', vertices)
  print '# Mountain contours: Plot by:'
  level_min= min(contours[0][0].level,contours[-1][0].level)
  level_max= max(contours[0][0].level,contours[-1][0].level)
  ps_per_lv= 3.9/(level_max-level_min)
  print '# 0th Mountain contours with variable-size points (ps=level):'
  print '''qplot -x /tmp/m_contours.dat u 1:2:'(0.1+{ps_per_lv}*($3-{level_min}))' index 0 w p pt 6 ps variable '''.format(ps_per_lv=ps_per_lv,level_min=level_min)
  print '# Mountain contours with lines (line color per mountain):'
  print '''bash -c 'p=""; for i in `seq 0 {nv}`;do p="$p /tmp/m_contours.dat index $i w l";done; qplot -x $p' '''.format(nv=len(vertices)-1)
  print '# Mountain contours with variable-size points (ps=level; point color per mountain):'
  print '''bash -c 'p=""; for i in `seq 0 {nv}`;do p="$p /tmp/m_contours.dat u 1:2:(0.1+{ps_per_lv}*(\$3-{level_min})) index $i w p pt 6 ps variable";done; qplot -x $p' '''.format(nv=len(vertices)-1,ps_per_lv=ps_per_lv,level_min=level_min)
  print '# i-th mountain contours (line color per level):'
  print '''bash -c 'i=0; p=""; for e in `seq 0 60`;do p="$p /tmp/m_contours.dat index $i ev :::$e::$e w l";done; qplot -x $p' '''

  #Approximate each contour in a mountain by an ellipse.
  print '# Approximate each contour in a mountain by an ellipse:'
  mountains= ApproxMountainByEllipses(vertices)
  WriteMountainEllipses('/tmp/m_ellipses.dat', mountains)
  print '# Ellipses: Plot by:'
  level_min= min(contours[0][0].level,contours[-1][0].level)
  level_max= max(contours[0][0].level,contours[-1][0].level)
  ps_per_lv= 3.9/(level_max-level_min)
  print '# Ellipses of mountain with lines (line color per mountain):'
  print '''bash -c 'p=""; for i in `seq 0 {nv}`;do p="$p /tmp/m_ellipses.dat index $i w l";done; qplot -x $p' '''.format(nv=len(vertices)-1)
  print '# First 10 ellipses of mountain with lines (line color per mountain):'
  print '''bash -c 'p=""; for i in `seq 0 {nv}`;do p="$p /tmp/m_ellipses.dat index $i ev :::0::10 w l";done; qplot -x $p' '''.format(nv=len(vertices)-1)
  print '# Ellipses of mountain with lines with data points (line color per mountain):'
  print '''bash -c 'p=""; for i in `seq 0 {nv}`;do p="$p /tmp/m_contours.dat index $i w dots lt $((i+1)) /tmp/m_ellipses.dat index $i w l lt $((i+1))";done; qplot -x $p' '''.format(nv=len(vertices)-1)
  print '# First 10 ellipses of mountain with lines with data points (line color per mountain):'
  print '''bash -c 'p=""; for i in `seq 0 {nv}`;do p="$p /tmp/m_contours.dat index $i w dots lt $((i+1)) /tmp/m_ellipses.dat index $i ev :::0::10 w l lt $((i+1))";done; qplot -x $p' '''.format(nv=len(vertices)-1)
  print '# Ellipses of mountain with variable-size points (ps=level; point color per mountain):'
  print '''bash -c 'p=""; for i in `seq 0 {nv}`;do p="$p /tmp/m_ellipses.dat u 1:2:(0.1+{ps_per_lv}*(\$3-{level_min})) index $i w p pt 6 ps variable";done; qplot -x $p' '''.format(nv=len(vertices)-1,ps_per_lv=ps_per_lv,level_min=level_min)
  print '# i-th ellipses of mountain (line color per level):'
  print '''bash -c 'i=0; p=""; for e in `seq 0 60`;do p="$p /tmp/m_ellipses.dat index $i ev :::$e::$e w l";done; qplot -x $p' '''
  print '# i-th ellipses of mountain with data points (line color per level):'
  print '''bash -c 'i=0; p=""; for e in `seq 0 10`;do p="$p /tmp/m_contours.dat index $i ev :::$e::$e w p lt $((e+1)) /tmp/m_ellipses.dat index $i ev :::$e::$e w l lt $((e+1))";done; qplot -x $p' '''

  #Convert mountains to a depth image.
  print '# Convert mountains to a depth image.'
  width,height= np.max(sum([[np.max(subcontour.points,0) for subcontour in subcontours] for subcontours in contours],[]),0)
  depth_img= MountainsToDepthImg(mountains, width, height)
  depth_img= cv2.cvtColor(depth_img.astype('uint8'), cv2.COLOR_GRAY2BGR)
  cv2.imwrite('/tmp/depth_img.png', depth_img)
  print '''display /tmp/depth_img.png'''
  cv2.imshow('depth', depth_img)
  #while cv2.waitKey() not in map(ord,[' ','q']):  pass

  #Make a valley map where a valley pixel is filled by its level.
  print '# Make a valley map where a valley pixel is filled by its level.'
  valley_img= MakeValleyMap(valleys, width, height)
  valley_img= cv2.cvtColor(valley_img.astype('uint8'), cv2.COLOR_GRAY2BGR)
  cv2.imwrite('/tmp/valley_img.png', valley_img)
  print '''display /tmp/valley_img.png'''
  cv2.imshow('valley', valley_img)
  while cv2.waitKey() not in map(ord,[' ','q']):  pass


