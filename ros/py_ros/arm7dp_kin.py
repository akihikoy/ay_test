#!/usr/bin/python
#\file    arm7dp_kin.py
#\brief   Kinematics of ode1/arm7_door_push_node.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Oct.29, 2015
from arm7dp_core import *
from geom_util import Transform, QFromAxisAngle, XToPosRot, InvRodrigues, Norm, AngleMod1, RandVec
import numpy as np
import numpy.linalg as la

def GetCOM(l):
  p_com= [0.0]*3
  m_total= 0.0
  for i in range(len(l.sensors.masses)):
    m_i= l.sensors.masses[i]
    x_i= l.sensors.link_x[7*i:7*i+7]
    p_com= [p_com[d]+m_i*x_i[d] for d in range(3)]
    m_total+= m_i
  p_com= [p_com[d]/m_total for d in range(3)]
  return p_com

#Detect singularity and fix it.
def ReviseJacobian(J, th=1.0e-3):
  for r in range(J.shape[0]):
    sqsum= [j*j for j in J[r,:]]
    if sqsum<th*th:  #r-th row is singular
      J[r,:]= 0.0
      print 'Revised J: r=%d, sqsum=%f' % (r,sqsum)

'''Compute a forward kinematics of an arm.
Return the gripper base pose on the base's frame.
  return: x; if not with_J; x: pose
  return: x, J; if with_J;  x: pose, J: Jacobian.
  l: l returned by Initialize.
  q: list of joint angles.
  x_ext: a local pose on gripper base frame.
    If not None, the returned pose is x_ext on the gripper base pose. '''
def FK(l, q, x_ext=None, with_J=False, viz=None):
  N= l.config.JointNum
  l_link= l.config.TotalArmLen/float(N+1)
  #x_unit= [0.0,0.0,0.0, 0.0,0.0,0.0,1.0]
  unit_axes= ((0.0,0.0,1.0), (1.0,0.0,0.0), (0.0,1.0,0.0))
  ax= lambda j: unit_axes[j%3]
  ztrans= lambda z: [0.0,0.0,z, 0.0,0.0,0.0,1.0]
  qtrans= lambda axis,theta: [0.0,0.0,0.0]+QFromAxisAngle(axis,theta).tolist()

  dT= [None]*(N+1)
  T= [None]*(N+1)
  dT[0]= Transform(ztrans(0.5*l.config.BaseLenZ + l.config.FSThick + l_link), qtrans(ax(0),q[0]))
  for j in range(1,N):
    dT[j]= Transform(ztrans(l_link), qtrans(ax(j),q[j]))
  dT[N]= ztrans(l_link + l.config.FSThick + 0.5*l.config.GBaseLenZ)  #gripper base
  T[1]= dT[0]
  if viz is not None:  viz(T[1])
  for j in range(1,N):
    T[j+1]= Transform(T[j],dT[j])
    if viz is not None:  viz(T[j+1])
  xe= Transform(T[N],dT[N])
  xe= xe if x_ext is None else Transform(xe,x_ext)
  if viz is not None:  viz(xe)

  if with_J:
    J= np.array([[0.0]*N for d in range(6)])
    pe= xe[:3]
    for j in range(N):
      p,R= XToPosRot(T[j+1])
      a= np.dot(R,ax(j))
      J[:3,j]= np.cross(a, pe-p)
      J[3:,j]= a
    return xe, J
  else:
    return xe

class TIKParam:
  def __init__(self):
    self.StepSize= 0.2
    self.Tolerance= 1.e-3
    self.MaxIteration= 200
    self.PoseErrWeight= 0.2
    self.SearchNoise= 1.0e-4
    self.QDiffPenalty= 1.0e-3
class TIKStatus:
  def __init__(self):
    self.IsSolved= None
    self.NumIter= None
    self.Error= None
    self.ErrorPos= None
    self.ErrorRot= None
    self.LastQ= None

'''Compute an inverse kinematics of an arm.
Return joint angles for a target gripper base pose on the base's frame.
  return: q, res;  q: joint angles (None if failure), res: IK status.
  l: l returned by Initialize.
  x_trg: target pose.
  x_ext: a local pose on gripper base frame.
    If not None, the returned q satisfies FK(l, q,x_ext)==x_trg.
  start_angles: initial joint angles for IK solver, or None (==l.sensors.joint_angles).
  with_st: whether return IK status. '''
def IK(l, x_trg, x_ext=None, start_angles=None, param=TIKParam(), with_st=False):
  if start_angles==None:  start_angles= l.sensors.joint_angles

  x_trg[3:]/= la.norm(x_trg[3:])  #Normalize the orientation:
  xw_trg= x_trg if x_ext==None else TransformRightInv(x_trg,x_ext)
  pw_trg,Rw_trg= XToPosRot(xw_trg)

  is_solved= False
  q= np.array(start_angles)
  for count in range(param.MaxIteration):
    x,J= FK(l, q, x_ext=x_ext, with_J=True)
    ReviseJacobian(J)
    p,R= XToPosRot(x)
    err_p= pw_trg-p
    err_R= np.dot(R, InvRodrigues(np.dot(R.T,Rw_trg)))
    err= np.concatenate(( err_p, param.PoseErrWeight*err_R ))
    if Norm(err) < param.Tolerance:
      is_solved= True
      break
    q+= param.StepSize*np.dot(la.pinv(J),err)
    q+= param.QDiffPenalty*(start_angles-q)  #Penalizing angles change
    q+= RandVec(q.shape[0],-param.SearchNoise,param.SearchNoise)  #Search noise to avoid singularity
    q= np.array(map(AngleMod1,q))
    #print Norm(err), Norm(err_p), Norm(err_R), '; ', q.tolist()
  q= np.array(map(AngleMod1,q))
  if with_st:
    st= TIKStatus()
    st.IsSolved= is_solved
    st.NumIter= count
    st.Error= Norm(err)
    st.ErrorPos= Norm(err_p)
    st.ErrorRot= Norm(err_R)
    st.LastQ= q
    return (q,st) if is_solved else (None,st)
  return q if is_solved else None


if __name__=='__main__':
  import random, math
  def Rand(xmin=-0.5,xmax=0.5):
    return random.random()*(xmax-xmin)+xmin

  #NOTE: sim is simulating: import joint_chain1 as sim
  sim= TContainer()
  sim.__dict__= globals()

  sim.rospy.init_node('arm7dp_kin')
  t,l= sim.Initialize()

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

  try:
    sim.SetupServiceProxy(t,l)
    sim.SetupPubSub(t,l)

    t.srvp.ode_resume()
    l.config= sim.GetConfig(t)
    print 'Current config=',l.config

    #Setup config
    l.config.JointNum= 7
    l.config.Box1Density1= 0.01
    l.config.Box1Density2= 0.5
    l.config.FixedBase= True
    l.config.ObjectMode= 0  #0: None, 1: Box1, ...

    #Reset to get state for plan
    sim.ResetConfig(t,l.config)
    time.sleep(0.1)  #Wait for l.sensors is updated
    print 'l.sensors=',l.sensors

    def FKTest():
      msg= sim.ode1.msg.ODEViz()
      x_base= l.sensors.link_x[0:7]
      sim.FK(l,l.sensors.joint_angles, x_ext=[0.0,0.0,0.05, 0.0,0.0,0.0,1.0], viz=lambda x:VizCube(l,Transform(x_base,x),msg))
      t.pub.ode_viz.publish(msg)

    #l.sensor_callback= lambda: FKTest()
    l.control_callback= FKTest

    x_dat= []

    theta0= l.sensors.joint_angles
    D= len(theta0)  #Number of joints
    for i in range(3):
      theta= [Rand(-math.pi,math.pi) for d in range(D)]
      sim.MoveToTheta(t,l,theta,dth_max=0.1)
      sim.SimSleep(t,l,0.5)
      x_dat.append(list(l.sensors.link_x[7:14]))

    def IKTest(x_trg):
      msg= sim.ode1.msg.ODEViz()
      x_base= l.sensors.link_x[0:7]
      VizCube(l,Transform(x_base,x_trg),msg)
      t.pub.ode_viz.publish(msg)

    #x_trg= [0.3,0.0,0.5, 0.0,0.0,0.0,1.0]
    for x_trg in x_dat:
      l.control_callback= lambda:IKTest(x_trg)
      q,ik_st= sim.IK(l, x_trg=x_trg, with_st=True)
      print ik_st.IsSolved, ik_st.NumIter, ik_st.Error, ik_st.ErrorPos, ik_st.ErrorRot
      if q is not None:
        sim.MoveToTheta(t,l,q,dth_max=0.1)
      else:
        print 'IK unsolved! Execute the last q.'
        sim.MoveToTheta(t,l,ik_st.LastQ,dth_max=0.1)
      sim.SimSleep(t,l,0.5)

  except Exception as e:
    sim.PrintException(e, ' in arm7dp_core.py')

  finally:
    sim.StopPubSub(t,l)
    t.srvp.ode_pause()

  sim.Cleanup(t)

