#!/usr/bin/python
#\file    arm7dp_chair1push.py
#\brief   Chair1 push.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Nov.09, 2015
import arm7dp_kin as sim
from geom_util import *
import time
import copy

#Visualize a box at end-link pose
def VizCube(l, x, msg=None, col=[0.4,1.0,0.4, 0.5], size=[0.08, 0.07, 0.04]):
  if msg is None:  msg= sim.ode1.msg.ODEViz()
  prm= sim.ode1.msg.ODEVizPrimitive()
  prm.type= prm.CUBE
  prm.pose= XToGPose(x)
  prm.param= size
  prm.color= sim.RGBA(*col)
  msg.objects.append(prm)
  return msg

def Viz(t,l):
  msg= sim.ode1.msg.ODEViz()
  x_base= l.sensors.link_x[0:7]
  VizCube(l,Transform(x_base,l.x_trg),msg)
  t.pub.ode_viz.publish(msg)

def KeyEventCallback(t,l,cmd):
  if cmd==ord('q'):    l.running= False

if __name__=='__main__':
  import random, math
  def Rand(xmin=-0.5,xmax=0.5):
    return random.random()*(xmax-xmin)+xmin

  sim.rospy.init_node('arm7dp_kin')
  t,l= sim.Initialize()

  try:
    sim.SetupServiceProxy(t,l)
    sim.SetupPubSub(t,l)

    t.srvp.ode_resume()
    l.config= sim.GetConfig(t)
    print 'Current config=',l.config

    #Setup config
    l.config.JointNum= 19
    l.config.TotalArmLen= 2.0
    l.config.Box1Density1= 0.01
    l.config.Box1Density2= 0.5
    l.config.FixedBase= True
    l.config.ObjectMode= 2  #0: None, 1: Box1, 2: Chair1, ...
    l.config.SliderFMax= 100

    #Reset to get state for plan
    sim.ResetConfig(t,l.config)
    time.sleep(0.1)  #Wait for l.sensors is updated
    print 'l.sensors=',l.sensors

    def IK2(x_trg,q_curr):
      ik_param= sim.TIKParam()
      ik_param.Tolerance= 1.0e-2
      ik_param.MaxIteration= 300
      q,ik_st= sim.IK(l, x_trg=x_trg, start_angles=q_curr, param=ik_param, with_st=True)
      if q is not None:
        return q
      else:
        print 'IK unsolved!'
        return ik_st.LastQ

    def MoveToX(t,l,x_trg,dth_max=0.1):
      if not l.running:  return
      l.x_trg= x_trg
      x_curr= sim.FK(l,l.sensors.joint_angles)
      x_traj= XInterpolation(x_curr,x_trg,10)
      q_traj= XTrajToQTraj(IK2, x_traj, start_angles=l.sensors.joint_angles)
      if q_traj==None:  return
      for q in q_traj:
        print q
        sim.MoveToTheta(t,l,q,dth_max=dth_max)
      #l.x_trg= copy.deepcopy(x_trg)
      #print ik_st.IsSolved, ik_st.NumIter, ik_st.Error, ik_st.ErrorPos, ik_st.ErrorRot
      #if q is not None:
        #sim.MoveToTheta(t,l,q,dth_max=dth_max)
      #else:
        #print 'IK unsolved!'
        ##print ik_st.LastQ
        #sim.MoveToTheta(t,l,ik_st.LastQ,dth_max=dth_max)

    #l.x_trg= sim.FK(l,l.sensors.joint_angles)
    #l.x_trg= [0.4,0.0,0.5, 0.0,0.0,0.0,1.0]
    def Push(t,l,angle=0.0):
      x_seat2= copy.deepcopy(l.sensors.chair1_x[14:21])
      q_way9= MultiplyQ(QFromAxisAngle([0.0,1.0,0.0],0.5*math.pi), QFromAxisAngle([1.0,0.0,0.0],0.5*math.pi))
      x_way9= [0.0,0.2,0.0]+q_way9.tolist()
      x_way8= copy.deepcopy(x_way9)
      x_way8[1]+= 0.5
      x_way7= copy.deepcopy(x_way8)
      x_way7[0]-= 0.4
      x_way6= copy.deepcopy(x_way7)
      x_way6[2]+= 0.2
      x_way10= copy.deepcopy(x_way9)
      x_way10[1]-= 0.1
      x_way10[3:]= MultiplyQ(QFromAxisAngle([0.0,0.0,1.0],angle), x_way10[3:])
      #x_way0=
      diff= DiffX(Transform(x_seat2, x_way9), l.sensors.link_x[7:14])[:3]
      print 'diff:',diff,Norm(diff)
      if Norm(diff)>0.2:
        MoveToX(t,l,x_trg=Transform(x_seat2, x_way6))
        MoveToX(t,l,x_trg=Transform(x_seat2, x_way7))
        MoveToX(t,l,x_trg=Transform(x_seat2, x_way8))
      MoveToX(t,l,x_trg=Transform(x_seat2, x_way9), dth_max=0.01)
      MoveToX(t,l,x_trg=Transform(x_seat2, x_way10), dth_max=0.01)

    l.running= True
    l.control_callback= lambda:Viz(t,l)
    l.keyevent_callback= lambda cmd:KeyEventCallback(t,l,cmd)

    x_trg0= [0.46, 0.51, 1.01, 0.0, 0.64421768723769246, 0.0, 0.76484218728448738]
    l.x_trg= x_trg0
    #MoveToX(t,l,x_trg=x_trg0)
    q0= [-0.055709324207661837, -4.3120723896642348, 4.2406183051659845, -1.9075692367310111, 0.45088662593823781, -0.08580670387895184, 0.60004593387815319, 0.65290188813532168, -0.44912719143156377, -0.98855959396181614, 0.69987245038631585, 0.22903898159863356, 0.73024427670610104, 0.37888215214709087, -0.16844872222619278, -0.93310075903268253, 0.051887689716982877, 0.19294297659330972, -0.24599250236168313]
    sim.MoveToTheta(t,l,q0,dth_max=0.1)

    angle= 0.0
    if l.running:  Push(t,l,angle)
    if l.running:  Push(t,l,angle)
    if l.running:  Push(t,l,angle)
    if l.running:  Push(t,l,angle)
    angle= 0.4
    if l.running:  Push(t,l,angle)
    if l.running:  Push(t,l,angle)
    if l.running:  Push(t,l,angle)
    if l.running:  Push(t,l,angle)
    angle= -0.4
    if l.running:  Push(t,l,angle)
    if l.running:  Push(t,l,angle)
    if l.running:  Push(t,l,angle)
    if l.running:  Push(t,l,angle)

    #sim.rospy.spin()
    #sim.rospy.signal_shutdown('Done.')

  except Exception as e:
    sim.PrintException(e, ' in arm7dp_core.py')

  finally:
    sim.StopPubSub(t,l)
    t.srvp.ode_pause()

  sim.Cleanup(t)
