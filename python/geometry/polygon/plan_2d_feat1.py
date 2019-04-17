#!/usr/bin/python
#\file    plan_2d_feat.py
#\brief   Convert grasp surface polygon to feature vector for learning.
#         Spline-based approach.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    May.29, 2017
import numpy as np
from splines.cubic_hermite_spline import TCubicHermiteSpline


def Main():
  #polygon is obtained by TGraspPlanningScene().Evaluate(p_grasp)['lps_ing_obj']
  polygon= [[0.02250000000000016, 0.03930382443834195], [-0.003290626669648233, 0.04959789330667163], [-0.021957423486490466, 0.049497413178730104], [-0.020668121534636333, 0.030722896579375318], [-0.022500000000000145, 0.02746341855873208], [-0.02250000000000013, 0.014768522919809221], [-0.004832145186925384, -0.0011258426224542365], [0.022500000000000013, -0.0020902717852315108]]
  #polygon= [[0.022500000000000006, -0.037541262786145405], [0.016516860487828647, -0.040081458863337624], [-0.008825324357486272, -0.03975948484903721], [-0.015582655826775522, -0.043317359413176965], [-0.022192838440299185, -0.04032478764029151], [-0.02249999999999998, -0.04052397240385208], [-0.022500000000000058, 0.03299817295938004], [-0.022026780623685238, 0.033122965387296324], [-0.012244817407781087, 0.02820430747657527], [0.012591800922777969, 0.038404424627968024], [0.019528341221562363, 0.03342603044074861], [0.02250000000000002, 0.03430389557517928]]
  #polygon= [[0.022499999999999992, -0.00025780267321809033], [0.012530831081980436, -0.0006733126321244896], [-0.0225, 0.004008395610238727], [-0.022500000000000013, -0.040533568494354716], [-0.013047025963346797, -0.03864946209662318], [0.007207952454395702, -0.022315577341018765], [0.022500000000000003, -0.04089476529274074]]
  #polygon= [[0.022499999999999614, 0.0069979261869305785], [-0.009138034214217827, -0.020794495160664228], [-0.00561141139444818, -0.028073564316844008], [0.011874683388431978, -0.03603613113390313], [0.01747772612823914, -0.04502330005314507], [0.02249999999999957, -0.04280791295683648]]
  #polygon= [[-0.022499999999999805, -0.03079293824807413], [0.008307701849726689, -0.041580311661123505], [0.022500000000000093, -0.049967419369178846], [0.02249999999999998, -0.01915320635587759], [0.013471528762861333, -0.024087509896815816], [0.0053348591199934405, 0.004261326065547697], [-0.021549132352616393, 0.016852845713883454], [-0.022499999999999735, 0.01666006672432078]]
  #polygon= [[0.022499999999999624, 0.014721580848192785], [-0.0036343473732008847, 0.015186657493631572], [-0.009644307461470731, 0.010635670607017901], [-0.0015802974183547992, -0.010471215149580809], [-0.005865750442236627, -0.01822698186042595], [0.009192379517686186, -0.02876601193335471], [0.022499999999999742, -0.030676780136362734]]
  #polygon= [[0.02250000000000016, 0.03930382443834195], [-0.003290626669648233, 0.04959789330667163], [-0.021957423486490466, 0.049497413178730104], [-0.020668121534636333, 0.030722896579375318], [-0.022500000000000145, 0.02746341855873208], [-0.02250000000000013, 0.014768522919809221], [-0.004832145186925384, -0.0011258426224542365], [0.022500000000000013, -0.0020902717852315108]]

  #Finger dimension:
  w_finger=[0.045, 0.02]
  y_mean= np.mean([p[1] for p in polygon])

  def arrange_kp(keypoints):
    if keypoints[0][0] > keypoints[-1][0]:  keypoints.reverse()
    keypoints2= [[-0.501*w_finger[0],y_mean]]
    if keypoints[0][0]-0.01*w_finger[0]>-0.501*w_finger[0]:
      keypoints2.append([keypoints[0][0]-0.01*w_finger[0],y_mean])
    for p in keypoints:
      if p[0]>keypoints2[-1][0]:  keypoints2.append(p)
    if keypoints[-1][0]+0.01*w_finger[0]<+0.501*w_finger[0]:
      keypoints2.append([keypoints[-1][0]+0.01*w_finger[0],y_mean])
    keypoints2.append([+0.501*w_finger[0],y_mean])
    return keypoints2

  keypoints1= filter(lambda p:p[1]>y_mean,polygon)
  keypoints1= arrange_kp(keypoints1)
  spline1= TCubicHermiteSpline()
  spline1.Initialize(keypoints1, tan_method=spline1.CARDINAL, c=1.0)
  feat1= [[x,spline1.Evaluate(x)] for x in np.mgrid[-0.499*w_finger[0]:0.499*w_finger[0]:9j]]

  keypoints2= filter(lambda p:p[1]<y_mean,polygon)
  keypoints2= arrange_kp(keypoints2)
  spline2= TCubicHermiteSpline()
  spline2.Initialize(keypoints2, tan_method=spline2.CARDINAL, c=1.0)
  feat2= [[x,spline2.Evaluate(x)] for x in np.mgrid[-0.499*w_finger[0]:0.499*w_finger[0]:9j]]

  sp1_viz= [[x,spline1.Evaluate(x)] for x in np.mgrid[-0.51*w_finger[0]:0.51*w_finger[0]:101j]]
  sp2_viz= [[x,spline2.Evaluate(x)] for x in np.mgrid[-0.51*w_finger[0]:0.51*w_finger[0]:101j]]

  def write_polygon(fp,polygon,closed=True):
    if len(polygon)>0:
      for pt in polygon+[polygon[0]] if closed else polygon:
        fp.write('%s\n'%' '.join(map(str,pt)))
    fp.write('\n')

  fp= open('/tmp/polygons.dat','w')
  write_polygon(fp,polygon)
  write_polygon(fp,sp1_viz,False)
  write_polygon(fp,sp2_viz,False)
  fp.close()
  fp= open('/tmp/feat.dat','w')
  write_polygon(fp,feat1,False)
  write_polygon(fp,feat2,False)
  fp.close()

def PlotGraphs():
  print 'Plotting graphs..'
  import os
  commands=[
    '''qplot -x2 aaa
        /tmp/polygons.dat u 1:2:'(column(-1)+1)' lc var w lp
        /tmp/feat.dat u 1:2:'(column(-1)+2)' ps 3 lc var w p
        &''',
        #/tmp/polygons.dat u 1:2:-1 lc var w l
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
