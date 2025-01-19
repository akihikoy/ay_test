#!/usr/bin/python3
#\file    plan_2d_grasp3.py
#\brief   Planning grasping on 2D.
#         Objects are given as polygons.
#         Gripper has two fingers that are rectangles.
#         Using plan_2d_grasp2.py, grasping parameter is planned with CMA-ES.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    May.19, 2017
from plan_2d_grasp2 import TGraspPlanningScene
import sys
sys.path.append('../')
import cma_es.cma as cma
import math
import numpy as np

def Main():
  contours= [
    [[0.729,0.049],[0.723,0.082],[0.702,0.125],[0.682,0.124],[0.654,0.106],[0.656,0.101],[0.647,0.081],[0.652,0.078],[0.651,0.071],[0.655,0.071],[0.673,0.031]],
    [[0.722,0.219],[0.717,0.220],[0.712,0.229],[0.693,0.235],[0.681,0.227],[0.672,0.230],[0.649,0.211],[0.637,0.213],[0.629,0.208],[0.626,0.216],[0.620,0.202],[0.616,0.203],[0.617,0.207],[0.609,0.200],[0.603,0.201],[0.601,0.191],[0.587,0.181],[0.589,0.175],[0.580,0.166],[0.585,0.133],[0.593,0.121],[0.605,0.113],[0.626,0.113],[0.645,0.121],[0.644,0.127],[0.651,0.123],[0.661,0.135],[0.669,0.134],[0.675,0.140],[0.702,0.148],[0.715,0.159],[0.717,0.150],[0.720,0.149],[0.721,0.167],[0.727,0.167],[0.730,0.195],[0.724,0.204]],
    [[0.820,0.156],[0.793,0.154],[0.812,0.154],[0.812,0.150],[0.803,0.149],[0.806,0.134],[0.802,0.139],[0.796,0.133],[0.786,0.140],[0.779,0.139],[0.772,0.131],[0.774,0.126],[0.782,0.127],[0.779,0.134],[0.789,0.130],[0.788,0.115],[0.794,0.109],[0.773,0.111],[0.769,0.124],[0.755,0.143],[0.749,0.144],[0.753,0.150],[0.750,0.153],[0.737,0.147],[0.731,0.149],[0.738,0.141],[0.722,0.144],[0.722,0.124],[0.726,0.126],[0.729,0.123],[0.725,0.118],[0.733,0.107],[0.733,0.090],[0.738,0.086],[0.738,0.077],[0.740,0.082],[0.744,0.080],[0.749,0.041],[0.757,0.039],[0.758,0.032],[0.763,0.034],[0.762,0.040],[0.769,0.037],[0.769,0.008],[0.781,0.024],[0.778,0.034],[0.788,0.043],[0.828,0.144],[0.819,0.150]],
    ]
  #for contour in contours:
    #print '['+','.join(['['+','.join(map(lambda f:'%0.3f'%f,p))+']' for p in contour])+'],'

  gps= TGraspPlanningScene()
  gps.Construct(contours, scale=0.9, obj_target=1)

  #ev= gps.Evaluate(p_grasp=[-0.01, 0.005, 0.1, 0.08])
  def f_obj(p_grasp, gps=gps):
    ev= gps.Evaluate(p_grasp)
    #Analyzing intersection polygon of inner-grasp polygon and target object in grasp local frame
    if len(ev['lps_ing_obj'])==0:  return None  #No grasp
    lps_ing_obj_max= np.max(ev['lps_ing_obj'],axis=0)
    lps_ing_obj_min= np.min(ev['lps_ing_obj'],axis=0)
    if lps_ing_obj_max[0]-lps_ing_obj_min[0]<gps.WFinger[0]*0.7:
      return None  #Grasping part is less than 70% of finger width.
    #if ev['area_ing_obj']==0.0:  return None  #Grasping area is zero
    score= 0.0
    collision= 10000.0*(sum(ev['area_f1_obj_s'])+sum(ev['area_f2_obj_s']))
    if collision>0.0:  score= 1.0+collision
    #score-= 10000.0*ev['area_ing_obj']
    return score
  cma_opt={
      'bounds': [[-0.1,-0.1,-math.pi,0.0], [0.1,0.1,math.pi,0.088]],
      'scaling_of_variables': [5.0,5.0,0.2,12.0],
      'verb_time': 0,
      'verb_log': False,
      'CMA_diagonal': 1,
      'maxfevals': 1000,
      'tolfun': 1.0e-4,
    }
  res= cma.fmin(f_obj, [0.0,0.0,0.0,0.08], 0.006, cma_opt)
  #print res
  p_grasp= res[0]
  print('Best p_grasp=',p_grasp)
  ev= gps.Evaluate(p_grasp)

  print('lps_ing_obj:',[a.tolist() for a in ev['lps_ing_obj']])

  #Compute areas of intersections:
  #NOTE: Use these values for evaluating collision.
  #Intersection with target object:
  print('Intersection area/target object: f1, f2=', end=' ')
  print(ev['area_f1_obj_s'][gps.ObjTarget], end=' ')
  print(ev['area_f2_obj_s'][gps.ObjTarget])
  #Intersection with other objects:
  for obj in range(len(gps.Contours)):
    if obj==gps.ObjTarget:  continue
    print('Intersection area/object',obj,': f1, f2=', end=' ')
    print(ev['area_f1_obj_s'][obj], end=' ')
    print(ev['area_f2_obj_s'][obj])

  print('Total collision area=',sum(ev['area_f1_obj_s'])+sum(ev['area_f2_obj_s']))

  #Save data into files for plotting:

  def write_polygon(fp,polygon):
    if len(polygon)>0:
      for pt in polygon+[polygon[0]]:
        fp.write('%s\n'%' '.join(map(str,pt)))
    fp.write('\n')

  fp= open('/tmp/contours.dat','w')
  #for contour in contours:
  for contour in gps.Contours:
    write_polygon(fp,contour)
  fp.close()

  fp= open('/tmp/viz.dat','w')
  fp.write('%s\n'%' '.join(map(str,gps.CenterO)))
  fp.write('%s\n'%' '.join(map(str,gps.CenterO+0.05*gps.ExO)))
  fp.write('\n')
  fp.write('%s\n'%' '.join(map(str,gps.CenterO)))
  fp.write('%s\n'%' '.join(map(str,gps.CenterO+0.025*gps.EyO)))
  fp.write('\n')
  write_polygon(fp,ev['ps_f1'])
  write_polygon(fp,ev['ps_f2'])
  write_polygon(fp,ev['ps_ing'])
  write_polygon(fp,ev['ps_ing_obj'])
  fp.close()

  fp= open('/tmp/intersection.dat','w')
  for ps_f1_obj in ev['ps_f1_obj_s']:
    write_polygon(fp,ps_f1_obj)
  for ps_f2_obj in ev['ps_f2_obj_s']:
    write_polygon(fp,ps_f2_obj)
  fp.close()

def PlotGraphs():
  print('Plotting graphs..')
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
      print('###',cmd)
      os.system(cmd)

  print('##########################')
  print('###Press enter to close###')
  print('##########################')
  input()
  os.system('qplot -x2kill aaa')

if __name__=='__main__':
  import sys
  if len(sys.argv)>1 and sys.argv[1] in ('p','plot','Plot','PLOT'):
    PlotGraphs()
    sys.exit(0)
  Main()
