#!/usr/bin/python
#\file    fk_gendata.py
#\brief   Forward kinematics learner: generate data.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.05, 2015
import joint_chain1 as sim
import time,sys,os,re
import math,random

def Rand(xmin=-0.5,xmax=0.5):
  return random.random()*(xmax-xmin)+xmin

#Visualize a box at end-link pose
def AddVizCube(msg,x,c=(0.4,1.0,0.4, 0.5),p=(0.08, 0.07, 0.04)):
  #msg= sim.ode1.msg.ODEViz()
  prm= sim.ode1.msg.ODEVizPrimitive()
  prm.type= prm.CUBE
  prm.pose= sim.XToGPose(x)
  prm.param= p
  prm.color= sim.RGBA(*c)
  msg.objects.append(prm)
  #t.pub.ode_viz.publish(msg)

#Visualize a box at end-link pose
def VizCubes(t,l):
  msg= sim.ode1.msg.ODEViz()
  for i in range(len(l.sensors.link_x)/7-1):
    AddVizCube(msg,l.sensors.link_x[7*i:7*(i+1)])
  AddVizCube(msg,l.sensors.link_x[-7:],p=(0.12, 0.10, 0.07))
  t.pub.ode_viz.publish(msg)

#Log and visualize
def LogViz(t,l):
  if l.c_samples>0:
    if l.c_skip==0:
      l.fp_q.write('%s\n'%(' '.join(map(str,l.sensors.joint_angles))))
      l.fp_x.write('%s\n'%(' '.join(map(str,l.sensors.link_x[-7:]))))
      l.fp_xall.write('%s\n'%(' '.join(map(str,l.sensors.link_x))))
      if l.c_samples%20==0:  print l.n_samples - l.c_samples, '/', l.n_samples
      l.c_samples-= 1
      l.c_skip= l.n_skip
    else:
      l.c_skip-= 1
  VizCubes(t,l)

def Main():
  sim.rospy.init_node('ros_min')
  t,l= sim.Initialize()

  sdofc= sys.argv[1] if len(sys.argv)>1 else '3'  #DoF code
  dof= int(re.search('^[0-9]+',sdofc).group())
  l.n_samples= int(sys.argv[2]) if len(sys.argv)>2 else 500
  l.n_skip= int(sys.argv[3]) if len(sys.argv)>3 else 2
  l.c_samples= l.n_samples
  l.c_skip= l.n_skip
  file_names= {'q':'datak/chain%s_q.dat'%sdofc,
               'x':'datak/chain%s_x.dat'%sdofc,
               'xall':'datak/chain%s_xall.dat'%sdofc}
  if os.path.exists(file_names['q']) or os.path.exists(file_names['x']) or os.path.exists(file_names['xall']):
    print 'File(s) already exists.'
    print 'Check:',file_names
    return
  l.fp_q= open(file_names['q'],'w')
  l.fp_x= open(file_names['x'],'w')
  l.fp_xall= open(file_names['xall'],'w')

  try:
    sim.SetupServiceProxy(t,l)
    sim.SetupPubSub(t,l)

    t.srvp.ode_resume()
    l.config= sim.GetConfig(t)
    #print 'Current config=',l.config

    #Setup config
    l.config.JointNum= dof

    #Reset to get state for plan
    sim.ResetConfig(t,l.config)
    time.sleep(0.1)  #Wait for l.sensors is updated
    #print 'l.sensors=',l.sensors

    l.sensor_callback= lambda: LogViz(t,l)

    while l.c_samples>0 and not sim.rospy.is_shutdown():
      theta= [Rand(-math.pi,math.pi) for d in range(l.config.JointNum)]
      sim.MoveToTheta(t,l,theta)

  except Exception as e:
    sim.PrintException(e, ' in fk_gendata.py')

  finally:
    sim.StopPubSub(t,l)
    t.srvp.ode_pause()

  sim.Cleanup(t)
  l.fp_q.close()
  l.fp_x.close()
  l.fp_xall.close()

  print 'Generated:',file_names

if __name__=='__main__':
  Main()
