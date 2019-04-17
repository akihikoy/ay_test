#!/usr/bin/python
#\file    plan_2d_grasp1.py
#\brief   Planning grasping on 2D.
#         Objects are given as polygons.
#         Gripper has two fingers that are rectangles.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    May.19, 2017
import math
import numpy as np
import numpy.linalg as la
from pca2 import TPCA
from polygon_clip import ClipPolygon
from polygon_area import PolygonArea

def Main():
  def Print(eq,g=globals(),l=locals()): print eq+'= '+str(eval(eq,g,l))

  contours= [
    [[0.729,0.049],[0.723,0.082],[0.702,0.125],[0.682,0.124],[0.654,0.106],[0.656,0.101],[0.647,0.081],[0.652,0.078],[0.651,0.071],[0.655,0.071],[0.673,0.031]],
    [[0.722,0.219],[0.717,0.220],[0.712,0.229],[0.693,0.235],[0.681,0.227],[0.672,0.230],[0.649,0.211],[0.637,0.213],[0.629,0.208],[0.626,0.216],[0.620,0.202],[0.616,0.203],[0.617,0.207],[0.609,0.200],[0.603,0.201],[0.601,0.191],[0.587,0.181],[0.589,0.175],[0.580,0.166],[0.585,0.133],[0.593,0.121],[0.605,0.113],[0.626,0.113],[0.645,0.121],[0.644,0.127],[0.651,0.123],[0.661,0.135],[0.669,0.134],[0.675,0.140],[0.702,0.148],[0.715,0.159],[0.717,0.150],[0.720,0.149],[0.721,0.167],[0.727,0.167],[0.730,0.195],[0.724,0.204]],
    [[0.820,0.156],[0.793,0.154],[0.812,0.154],[0.812,0.150],[0.803,0.149],[0.806,0.134],[0.802,0.139],[0.796,0.133],[0.786,0.140],[0.779,0.139],[0.772,0.131],[0.774,0.126],[0.782,0.127],[0.779,0.134],[0.789,0.130],[0.788,0.115],[0.794,0.109],[0.773,0.111],[0.769,0.124],[0.755,0.143],[0.749,0.144],[0.753,0.150],[0.750,0.153],[0.737,0.147],[0.731,0.149],[0.738,0.141],[0.722,0.144],[0.722,0.124],[0.726,0.126],[0.729,0.123],[0.725,0.118],[0.733,0.107],[0.733,0.090],[0.738,0.086],[0.738,0.077],[0.740,0.082],[0.744,0.080],[0.749,0.041],[0.757,0.039],[0.758,0.032],[0.763,0.034],[0.762,0.040],[0.769,0.037],[0.769,0.008],[0.781,0.024],[0.778,0.034],[0.788,0.043],[0.828,0.144],[0.819,0.150]],
    ]
  #for contour in contours:
    #print '['+','.join(['['+','.join(map(lambda f:'%0.3f'%f,p))+']' for p in contour])+'],'

  #Target object to be grasped:
  obj_target= 1

  #Defining object local frame of target object:
  pca= TPCA(contours[obj_target])
  center_o= pca.Mean
  ex_o,ey_o= pca.EVecs[:,0], pca.EVecs[:,1]
  theta_o= math.atan2(ex_o[1],ex_o[0])

  #Grasping parameters: [x,y,theta,w0],
  #where [x,y,theta] is a local pose in object frame, w0 is grasping width.
  p_grasp= [-0.01, 0.005, 0.1, 0.08]
  w_gripper= [0.045, 0.02]  #Gripper width (horizontal, vertical)

  #Making finger polygons for collision check:
  c_g= center_o + p_grasp[0]*ex_o + p_grasp[1]*ey_o  #Center of grasp
  th= theta_o + p_grasp[2]
  R_g= np.array([[math.cos(th),-math.sin(th)],[math.sin(th),math.cos(th)]])  #Rot matrix of grasp
  #note: np.dot(R_g,p) == (np.dot(R_g,p.T)).T
  w0= p_grasp[3]
  #Local finger polygons:
  lps_f1= [[0.5*w_gripper[0],0.5*w0], [0.5*w_gripper[0],0.5*w0+w_gripper[1]],
           [-0.5*w_gripper[0],0.5*w0+w_gripper[1]], [-0.5*w_gripper[0],0.5*w0]]
  lps_f2= [[0.5*w_gripper[0],-0.5*w0], [0.5*w_gripper[0],-0.5*w0-w_gripper[1]],
           [-0.5*w_gripper[0],-0.5*w0-w_gripper[1]], [-0.5*w_gripper[0],-0.5*w0]]
  #Global finger polygons:
  ps_f1= [c_g+np.dot(R_g,lp) for lp in lps_f1]
  ps_f2= [c_g+np.dot(R_g,lp) for lp in lps_f2]

  #Making inner-grasp polygon for grasp quality:
  lps_ing= [[0.5*w_gripper[0],0.5*w0], [0.5*w_gripper[0],-0.5*w0],
            [-0.5*w_gripper[0],-0.5*w0], [-0.5*w_gripper[0],0.5*w0]]
  ps_ing= [c_g+np.dot(R_g,lp) for lp in lps_ing]

  #Intersection of inner-grasp polygon and target object:
  ps_ing_obj= ClipPolygon(contours[obj_target], ps_ing)
  #Convert to grasp local frame:
  lps_ing_obj= [np.dot(R_g.T,p-c_g) for p in ps_ing_obj]  #NOTE: Use this as a contact surface model.
  Print('lps_ing_obj',l=locals())

  #Intersection of finger polygons and objects:
  ps_f1_obj_s= []
  ps_f2_obj_s= []
  for contour in contours:
    ps_f1_obj_s.append(ClipPolygon(contour, ps_f1))
    ps_f2_obj_s.append(ClipPolygon(contour, ps_f2))

  #Compute areas of intersections:
  #NOTE: Use these values for evaluating collision.
  #Intersection with target object:
  print 'Intersection area/target object: f1, f2=',
  print PolygonArea(ps_f1_obj_s[obj_target]),
  print PolygonArea(ps_f2_obj_s[obj_target])
  #Intersection with other objects:
  for obj in range(len(contours)):
    if obj==obj_target:  continue
    print 'Intersection area/object',obj,': f1, f2=',
    print PolygonArea(ps_f1_obj_s[obj]),
    print PolygonArea(ps_f2_obj_s[obj])


  #Save data into files for plotting:

  def write_polygon(fp,polygon):
    if len(polygon)>0:
      for pt in polygon+[polygon[0]]:
        fp.write('%s\n'%' '.join(map(str,pt)))
    fp.write('\n')

  fp= open('/tmp/contours.dat','w')
  for contour in contours:
    write_polygon(fp,contour)
  fp.close()

  fp= open('/tmp/viz.dat','w')
  fp.write('%s\n'%' '.join(map(str,center_o)))
  fp.write('%s\n'%' '.join(map(str,center_o+0.05*ex_o)))
  fp.write('\n')
  fp.write('%s\n'%' '.join(map(str,center_o)))
  fp.write('%s\n'%' '.join(map(str,center_o+0.025*ey_o)))
  fp.write('\n')
  write_polygon(fp,ps_f1)
  write_polygon(fp,ps_f2)
  write_polygon(fp,ps_ing)
  write_polygon(fp,ps_ing_obj)
  fp.close()

  fp= open('/tmp/intersection.dat','w')
  for ps_f1_obj in ps_f1_obj_s:
    write_polygon(fp,ps_f1_obj)
  for ps_f2_obj in ps_f2_obj_s:
    write_polygon(fp,ps_f2_obj)
  fp.close()

def PlotGraphs():
  print 'Plotting graphs..'
  import os
  commands=[
    '''qplot -x2 aaa
        /tmp/contours.dat w l
        /tmp/viz.dat u 1:2:'(column(-1)+1)' lc var w l
        /tmp/intersection.dat w l
        &''',
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
