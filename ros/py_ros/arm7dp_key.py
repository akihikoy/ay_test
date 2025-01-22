#!/usr/bin/python3
#\file    arm7dp_key.py
#\brief   End-effector control with key input for ode1/arm7_door_push_node.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Oct.31, 2015
import arm7dp_kin as sim
from geom_util import Transform, QFromAxisAngle, XToGPose
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
  elif cmd==ord(','):  l.x_trg[0]+= 0.01
  elif cmd==ord('.'):  l.x_trg[0]-= 0.01
  elif cmd==ord('k'):  l.x_trg[1]+= 0.01
  elif cmd==ord('l'):  l.x_trg[1]-= 0.01
  elif cmd==ord(';'):  l.x_trg[2]+= 0.01
  elif cmd==ord('/'):  l.x_trg[2]-= 0.01
  elif cmd==ord('<'):  l.x_trg= l.x_trg[:3]+Transform(QFromAxisAngle((0.0,1.0,0.0),0.1),l.x_trg[3:]).tolist()
  elif cmd==ord('>'):  l.x_trg= l.x_trg[:3]+Transform(QFromAxisAngle((0.0,1.0,0.0),-0.1),l.x_trg[3:]).tolist()

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
    print('Current config=',l.config)

    #Setup config
    l.config.JointNum= 15
    l.config.TotalArmLen= 2.0
    l.config.Box1Density1= 0.01
    l.config.Box1Density2= 0.5
    l.config.FixedBase= True
    l.config.ObjectMode= 2  #0: None, 1: Box1, 2: Chair1, ...

    #Reset to get state for plan
    sim.ResetConfig(t,l.config)
    time.sleep(0.1)  #Wait for l.sensors is updated
    print('l.sensors=',l.sensors)

    #l.x_trg= sim.FK(l,l.sensors.joint_angles)
    l.x_trg= [0.4,0.0,0.5, 0.0,0.0,0.0,1.0]
    l.x_trg_prev= copy.deepcopy(l.x_trg)
    l.running= True

    l.control_callback= lambda:Viz(t,l)
    l.keyevent_callback= lambda cmd:KeyEventCallback(t,l,cmd)
    ik_param= sim.TIKParam()
    ik_param.Tolerance= 1.0e-2
    #ik_param.MaxIteration= 200

    while l.running:
      q,ik_st= sim.IK(l, x_trg=l.x_trg, param=ik_param, with_st=True)
      print(ik_st.IsSolved, ik_st.NumIter, ik_st.Error, ik_st.ErrorPos, ik_st.ErrorRot)
      if q is not None:
        l.x_trg_prev= copy.deepcopy(l.x_trg)
        print(l.x_trg, q)
        sim.MoveToTheta(t,l,q,dth_max=0.1)
      else:
        print('IK unsolved!')
        sim.MoveToTheta(t,l,ik_st.LastQ,dth_max=0.1)
        print(l.x_trg, ik_st.LastQ)
        #print l.sensors.joint_angles
        l.x_trg= copy.deepcopy(l.x_trg_prev)

    #sim.rospy.spin()
    #sim.rospy.signal_shutdown('Done.')

  except Exception as e:
    sim.PrintException(e, ' in arm7dp_core.py')

  finally:
    sim.StopPubSub(t,l)
    t.srvp.ode_pause()

  sim.Cleanup(t)
