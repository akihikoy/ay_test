#!/usr/bin/python
#\file    plan_2d_grasp2a.py
#\brief   Planning grasping on 2D.
#         Objects are given as polygons.
#         Gripper has two fingers that are rectangles.
#         Based on plan_2d_grasp2.py, improvements based on experiments are made.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jun.07, 2017
import math
import numpy as np
import numpy.linalg as la
from pca2 import TPCA
from polygon_clip import ClipPolygon
from polygon_area import PolygonArea
from polygon_shrink import ShrinkPolygon

class TGraspPlanningScene(object):
  #w_finger: Finger dimension (Horizontal length, Vertical length).
  def __init__(self, w_finger=[0.045, 0.02]):
    self.WFinger= w_finger

  #Construct a planning scene.
  #  contours: contours (set of sequences of points) each of that represents an object.
  #  scale: If not 1, each contour is scaled.
  def Construct(self, contours, scale=1.0, obj_target=None):
    if scale==1.0:  self.Contours= contours
    else:  self.Contours= [ShrinkPolygon(contour,scale) for contour in contours]
    if obj_target is not None:
      self.SetTarget(obj_target)

  #Set a target object by index of contours.
  def SetTarget(self, obj_target):
    self.ObjTarget= obj_target

    #Defining object local frame of target object:
    pca= TPCA(self.Contours[self.ObjTarget])
    self.CenterO= pca.Mean
    self.ExO,self.EyO= pca.EVecs[:,0], pca.EVecs[:,1]
    self.ThetaO= math.atan2(self.ExO[1],self.ExO[0])

  '''Evaluate a grasp parameter p_grasp.
      Grasping parameters: p_grasp=[x,y,theta,w0],
      where [x,y,theta] is a local pose in object target frame, w0 is initial gripper width. '''
  def Evaluate(self, p_grasp):
    #Making finger polygons for evaluation:
    #Global grasp pose:
    c_g= self.CenterO + p_grasp[0]*self.ExO + p_grasp[1]*self.EyO  #Center of grasp
    th_g= self.ThetaO + p_grasp[2]
    R_g= np.array([[math.cos(th_g),-math.sin(th_g)],[math.sin(th_g),math.cos(th_g)]])  #Rot matrix of grasp
    #note: np.dot(R_g,p) == (np.dot(R_g,p.T)).T
    w0= p_grasp[3]
    #Local finger polygons in target object frame:
    lps_f1= [[0.5*self.WFinger[0],0.5*w0], [0.5*self.WFinger[0],0.5*w0+self.WFinger[1]],
            [-0.5*self.WFinger[0],0.5*w0+self.WFinger[1]], [-0.5*self.WFinger[0],0.5*w0]]
    lps_f2= [[0.5*self.WFinger[0],-0.5*w0], [0.5*self.WFinger[0],-0.5*w0-self.WFinger[1]],
            [-0.5*self.WFinger[0],-0.5*w0-self.WFinger[1]], [-0.5*self.WFinger[0],-0.5*w0]]
    #Global finger polygons:
    ps_f1= [c_g+np.dot(R_g,lp) for lp in lps_f1]
    ps_f2= [c_g+np.dot(R_g,lp) for lp in lps_f2]

    #Making inner-grasp polygon for grasp quality:
    #lps_ing= [[0.5*self.WFinger[0],0.5*w0], [0.5*self.WFinger[0],-0.5*w0],
              #[-0.5*self.WFinger[0],-0.5*w0], [-0.5*self.WFinger[0],0.5*w0]]
    lps_ing= [[0.5*self.WFinger[0],0.5*w0+self.WFinger[1]], [0.5*self.WFinger[0],-0.5*w0-self.WFinger[1]],
              [-0.5*self.WFinger[0],-0.5*w0-self.WFinger[1]], [-0.5*self.WFinger[0],0.5*w0+self.WFinger[1]]]
    ps_ing= [c_g+np.dot(R_g,lp) for lp in lps_ing]

    #Intersection of inner-grasp polygon and target object:
    ps_ing_obj= ClipPolygon(self.Contours[self.ObjTarget], ps_ing)
    #Convert to grasp local frame:
    lps_ing_obj= [np.dot(R_g.T,p-c_g) for p in ps_ing_obj]  #NOTE: Use this as a contact surface model.
    #Its area:
    area_ing_obj= PolygonArea(ps_ing_obj)

    #Intersection of finger polygons and objects:
    ps_f1_obj_s= [ClipPolygon(contour, ps_f1) for contour in self.Contours]
    ps_f2_obj_s= [ClipPolygon(contour, ps_f2) for contour in self.Contours]

    #Compute areas of intersections:
    #NOTE: Use these values for evaluating collision.
    area_f1_obj_s= [PolygonArea(polygon) for polygon in ps_f1_obj_s]
    area_f2_obj_s= [PolygonArea(polygon) for polygon in ps_f2_obj_s]

    return {
        'c_g'          :c_g          ,  #Grasp center in global frame
        'th_g'         :th_g         ,  #Grasp orientation in global frame
        'w0'           :w0           ,  #Initial gripper width
        'ps_f1'        :ps_f1        ,  #Finger polygons in global frame
        'ps_f2'        :ps_f2        ,  #Finger polygons in global frame
        'ps_ing'       :ps_ing       ,  #Inner-grasp polygon in global frame
        'ps_ing_obj'   :ps_ing_obj   ,  #Intersection polygon of inner-grasp polygon and target object in global frame
        'lps_ing_obj'  :lps_ing_obj  ,  #Intersection polygon of inner-grasp polygon and target object in grasp local frame
        'area_ing_obj' :area_ing_obj ,  #Area of ps_ing_obj
        'ps_f1_obj_s'  :ps_f1_obj_s  ,  #Intersection polygons of finger polygon and objects
        'ps_f2_obj_s'  :ps_f2_obj_s  ,  #Intersection polygons of finger polygon and objects
        'area_f1_obj_s':area_f1_obj_s,  #Areas of ps_f1_obj_s
        'area_f2_obj_s':area_f2_obj_s,  #Areas of ps_f1_obj_s
      }

def Main():
  contours= [
    [[0.729,0.049],[0.723,0.082],[0.702,0.125],[0.682,0.124],[0.654,0.106],[0.656,0.101],[0.647,0.081],[0.652,0.078],[0.651,0.071],[0.655,0.071],[0.673,0.031]],
    [[0.722,0.219],[0.717,0.220],[0.712,0.229],[0.693,0.235],[0.681,0.227],[0.672,0.230],[0.649,0.211],[0.637,0.213],[0.629,0.208],[0.626,0.216],[0.620,0.202],[0.616,0.203],[0.617,0.207],[0.609,0.200],[0.603,0.201],[0.601,0.191],[0.587,0.181],[0.589,0.175],[0.580,0.166],[0.585,0.133],[0.593,0.121],[0.605,0.113],[0.626,0.113],[0.645,0.121],[0.644,0.127],[0.651,0.123],[0.661,0.135],[0.669,0.134],[0.675,0.140],[0.702,0.148],[0.715,0.159],[0.717,0.150],[0.720,0.149],[0.721,0.167],[0.727,0.167],[0.730,0.195],[0.724,0.204]],
    [[0.820,0.156],[0.793,0.154],[0.812,0.154],[0.812,0.150],[0.803,0.149],[0.806,0.134],[0.802,0.139],[0.796,0.133],[0.786,0.140],[0.779,0.139],[0.772,0.131],[0.774,0.126],[0.782,0.127],[0.779,0.134],[0.789,0.130],[0.788,0.115],[0.794,0.109],[0.773,0.111],[0.769,0.124],[0.755,0.143],[0.749,0.144],[0.753,0.150],[0.750,0.153],[0.737,0.147],[0.731,0.149],[0.738,0.141],[0.722,0.144],[0.722,0.124],[0.726,0.126],[0.729,0.123],[0.725,0.118],[0.733,0.107],[0.733,0.090],[0.738,0.086],[0.738,0.077],[0.740,0.082],[0.744,0.080],[0.749,0.041],[0.757,0.039],[0.758,0.032],[0.763,0.034],[0.762,0.040],[0.769,0.037],[0.769,0.008],[0.781,0.024],[0.778,0.034],[0.788,0.043],[0.828,0.144],[0.819,0.150]],
    ]
  #for contour in contours:
    #print '['+','.join(['['+','.join(map(lambda f:'%0.3f'%f,p))+']' for p in contour])+'],'

  gps= TGraspPlanningScene()
  gps.Construct(contours, obj_target=1)

  #ev= gps.Evaluate(p_grasp=[-0.01, 0.005, 0.1, 0.08])
  ev= gps.Evaluate(p_grasp=[0.06, 0.005, -0.2, 0.07])

  print 'lps_ing_obj:',ev['lps_ing_obj']

  #Compute areas of intersections:
  #NOTE: Use these values for evaluating collision.
  #Intersection with target object:
  print 'Intersection area/target object: f1, f2=',
  print ev['area_f1_obj_s'][gps.ObjTarget],
  print ev['area_f2_obj_s'][gps.ObjTarget]
  #Intersection with other objects:
  for obj in range(len(gps.Contours)):
    if obj==gps.ObjTarget:  continue
    print 'Intersection area/object',obj,': f1, f2=',
    print ev['area_f1_obj_s'][obj],
    print ev['area_f2_obj_s'][obj]

  print 'Total collision area=',sum(ev['area_f1_obj_s'])+sum(ev['area_f2_obj_s'])

  #Save data into files for plotting:

  def write_polygon(fp,polygon):
    if len(polygon)>0:
      for pt in polygon+[polygon[0]]:
        fp.write('%s\n'%' '.join(map(str,pt)))
    fp.write('\n')

  with open('/tmp/contours.dat','w') as fp:
    for contour in contours:
      write_polygon(fp,contour)

  #with open('/tmp/viz.dat','w') as fp:
  with open('/tmp/viz-ax.dat','w') as fp:
    fp.write('%s\n'%' '.join(map(str,gps.CenterO)))
    fp.write('%s\n'%' '.join(map(str,gps.CenterO+0.05*gps.ExO)))
    fp.write('\n')
    fp.write('%s\n'%' '.join(map(str,gps.CenterO)))
    fp.write('%s\n'%' '.join(map(str,gps.CenterO+0.025*gps.EyO)))
    fp.write('\n')
  with open('/tmp/viz-ps_f.dat','w') as fp:
    write_polygon(fp,ev['ps_f1'])
    write_polygon(fp,ev['ps_f2'])
  with open('/tmp/viz-ps_ing.dat','w') as fp:
    write_polygon(fp,ev['ps_ing'])
  with open('/tmp/viz-ps_ing_obj.dat','w') as fp:
    write_polygon(fp,ev['ps_ing_obj'])
  with open('/tmp/viz-lps_ing_obj.dat','w') as fp:
    write_polygon(fp,ev['lps_ing_obj'])

  with open('/tmp/intersection.dat','w') as fp:
    for ps_f1_obj in ev['ps_f1_obj_s']:
      write_polygon(fp,ps_f1_obj)
    for ps_f2_obj in ev['ps_f2_obj_s']:
      write_polygon(fp,ps_f2_obj)

def PlotGraphs():
  print 'Plotting graphs..'
  import os
  commands=[
    '''qplot -x2 aaa
        -s 'set size ratio -1;'
        /tmp/contours.dat w l
        /tmp/viz-ax.dat w l
        /tmp/viz-ps_f.dat w l
        /tmp/viz-ps_ing.dat w l
        /tmp/viz-ps_ing_obj.dat w l
        /tmp/intersection.dat w l
        &''',
        # -o plan_2d_grasp2a-1.svg
        #/tmp/viz.dat u 1:2:'(column(-1)+1)' lc var w l
        #/tmp/viz.dat u 1:2:-1 lc var w l
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
