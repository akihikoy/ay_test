#!/usr/bin/python
#\file    joint_chain1.py
#\brief   Using ode1/joint_chain1_node.
#         cf. pr2_lfd_trick/core_tool.py, m_tsim_core1.phy
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.05, 2015
import roslib; roslib.load_manifest('ode1')
import rospy
import std_msgs.msg
import std_srvs.srv
import geometry_msgs.msg
import ode1.msg
import ode1.srv
import sys
import time
import traceback
import numpy as np

#Container class that can hold any variables
#ref. http://blog.beanz-net.jp/happy_programming/2008/11/python-5.html
class TContainer:
  def __init__(self,debug=False):
    #self._debug= debug
    #if self._debug:
    if debug:  print 'Created TContainer object',hex(id(self))
  def __del__(self):
    #if self._debug:
    print 'Deleting TContainer object',hex(id(self))
  def __str__(self):
    return str(self.__dict__)
  def __repr__(self):
    return str(self.__dict__)
  def __iter__(self):
    return self.__dict__.itervalues()
  def items(self):
    return self.__dict__.items()
  def iteritems(self):
    return self.__dict__.iteritems()
  def keys(self):
    return self.__dict__.keys()
  def values(self):
    return self.__dict__.values()
  def __getitem__(self,key):
    return self.__dict__[key]
  def __setitem__(self,key,value):
    self.__dict__[key]= value
  def __delitem__(self,key):
    del self.__dict__[key]
  def __contains__(self,key):
    return key in self.__dict__
  def Cleanup(self):
    keys= self.__dict__.keys()
    for k in keys:
      if k!='_debug':
        self.__dict__[k]= None
        del self.__dict__[k]

#Print an exception with a good format
def PrintException(e, msg=''):
  c1= ''
  c2= ''
  c3= ''
  ce= ''
  print '%sException( %s%r %s)%s:' % (c1, c2,type(e), c1, msg)
  print '%r' % (e)
  print '  %sTraceback: ' % (c3)
  print '{'
  traceback.print_tb(sys.exc_info()[2])
  print '}'
  print '%s# Exception( %s%r %s)%s:' % (c1, c2,type(e), c1, msg)
  print '# %r%s' % (e, ce)

#Return std_msgs/ColorRGBA
def RGBA(r,g,b,a):
  color= std_msgs.msg.ColorRGBA()
  color.r= r
  color.g= g
  color.b= b
  color.a= a
  return color

#Convert x to geometry_msgs/Pose
def XToGPose(x):
  pose= geometry_msgs.msg.Pose()
  pose.position.x= x[0]
  pose.position.y= x[1]
  pose.position.z= x[2]
  pose.orientation.x= x[3]
  pose.orientation.y= x[4]
  pose.orientation.z= x[5]
  pose.orientation.w= x[6]
  return pose

#Convert geometry_msgs/Pose to x
def GPoseToX(pose):
  x= [0]*7
  x[0]= pose.position.x
  x[1]= pose.position.y
  x[2]= pose.position.z
  x[3]= pose.orientation.x
  x[4]= pose.orientation.y
  x[5]= pose.orientation.z
  x[6]= pose.orientation.w
  return x

#t is a dummy of core_tool.TCoreTool
def Initialize():
  t= TContainer(debug=True)
  t.pub= TContainer(debug=True)  #Publishers
  t.sub= TContainer(debug=True)  #Subscribers
  t.srvp= TContainer(debug=True)  #Service proxies
  l= TContainer(debug=True)
  return t,l

def Cleanup(t):
  for k in t.sub.keys():
    print 'Stop subscribing %r...' % k,
    t.sub[k].unregister()
    del t.sub[k]
    print 'ok'

  for k in t.pub.keys():
    print 'Stop publishing %r...' % k,
    t.pub[k].unregister()
    del t.pub[k]
    print 'ok'

  for k in t.srvp.keys():
    print 'Delete service proxy %r...' % k,
    del t.srvp[k]
    print 'ok'

def SetupServiceProxy(t,l):
  if 'ode_get_config' not in t.srvp:
    print 'Waiting for /chain_sim/get_config...'
    rospy.wait_for_service('/chain_sim/get_config',3.0)
    t.srvp.ode_get_config= rospy.ServiceProxy('/chain_sim/get_config', ode1.srv.ODEGetConfig, persistent=False)
  if 'ode_reset2' not in t.srvp:
    print 'Waiting for /chain_sim/reset2...'
    rospy.wait_for_service('/chain_sim/reset2',3.0)
    t.srvp.ode_reset2= rospy.ServiceProxy('/chain_sim/reset2', ode1.srv.ODEReset2, persistent=False)
  if 'ode_pause' not in t.srvp:
    print 'Waiting for /chain_sim/pause...'
    rospy.wait_for_service('/chain_sim/pause',3.0)
    t.srvp.ode_pause= rospy.ServiceProxy('/chain_sim/pause', std_srvs.srv.Empty, persistent=False)
  if 'ode_resume' not in t.srvp:
    print 'Waiting for /chain_sim/resume...'
    rospy.wait_for_service('/chain_sim/resume',3.0)
    t.srvp.ode_resume= rospy.ServiceProxy('/chain_sim/resume', std_srvs.srv.Empty, persistent=False)

def SetupPubSub(t,l):
  StopPubSub(t,l)
  if 'ode_theta' not in t.pub:
    t.pub.ode_theta= rospy.Publisher("/chain_sim/theta", std_msgs.msg.Float64MultiArray)
  if 'ode_viz' not in t.pub:
    t.pub.ode_viz= rospy.Publisher("/chain_sim/viz", ode1.msg.ODEViz)
  if 'ode_sensors' not in t.sub:
    t.sub.ode_sensors= rospy.Subscriber("/chain_sim/sensors", ode1.msg.ODESensor, lambda msg:ODESensorCallback(msg,t,l))
  if 'sensor_callback' not in l:
    l.sensor_callback= None
  if 'control_callback' not in l:
    l.control_callback= None

def StopPubSub(t,l):
  if 'ode_sensors' in t.sub:
    t.sub.ode_sensors.unregister()
    del t.sub.ode_sensors

def ODESensorCallback(msg,t,l):
  l.sensors= msg
  if l.sensor_callback!=None:
    l.sensor_callback()

def GetConfig(t):
  return t.srvp.ode_get_config().config

def ResetConfig(t,config):
  t.pub.ode_viz.publish(ode1.msg.ODEViz())  #Clear visualization
  req= ode1.srv.ODEReset2Request()
  req.config= config
  t.srvp.ode_reset2(req)

def SimSleep(t,l,dt):
  tc0= l.sensors.time
  while l.sensors.time-tc0<dt:
    time.sleep(dt*0.02)

def MoveDTheta(t,l,dth):
  dt= l.config.TimeStep
  theta0= l.sensors.joint_angles
  theta_msg= std_msgs.msg.Float64MultiArray()
  theta_msg.data= np.array(theta0) + dth
  t.pub.ode_theta.publish(theta_msg)
  SimSleep(t,l,dt)
  if l.control_callback!=None:
    l.control_callback()

def MoveToTheta(t,l,th,dth_max=0.5):
  cnt= True
  while cnt and not rospy.is_shutdown():
    theta= l.sensors.joint_angles
    dth= np.array(th) - theta
    dth0= max(map(abs,dth))
    if dth0>dth_max:  dth*= dth_max/dth0
    else:  cnt= False
    MoveDTheta(t,l,dth)


if __name__=='__main__':
  import random, math
  def Rand(xmin=-0.5,xmax=0.5):
    return random.random()*(xmax-xmin)+xmin

  #NOTE: sim is simulating: import joint_chain1 as sim
  sim= TContainer()
  sim.__dict__= globals()

  sim.rospy.init_node('ros_min')
  t,l= sim.Initialize()

  #Visualize a box at end-link pose
  def VizCube(t,l):
    msg= sim.ode1.msg.ODEViz()
    prm= sim.ode1.msg.ODEVizPrimitive()
    prm.type= prm.CUBE
    prm.pose= sim.XToGPose(l.sensors.link_x[-7:])
    prm.param= [0.08, 0.07, 0.04]
    prm.color= sim.RGBA(0.4,1.0,0.4, 0.5)
    msg.objects.append(prm)
    t.pub.ode_viz.publish(msg)

  try:
    sim.SetupServiceProxy(t,l)
    sim.SetupPubSub(t,l)

    t.srvp.ode_resume()
    l.config= sim.GetConfig(t)
    print 'Current config=',l.config

    #Setup config
    l.config.JointNum= 3

    #Reset to get state for plan
    sim.ResetConfig(t,l.config)
    time.sleep(0.1)  #Wait for l.sensors is updated
    print 'l.sensors=',l.sensors

    #l.sensor_callback= lambda: VizCube(t,l)

    #Control arm
    theta0= l.sensors.joint_angles
    D= len(theta0)  #Number of joints
    for i in range(200):
      dth= [0.0]*D
      dth[(i/5)%D]= 0.1
      sim.MoveDTheta(t,l,dth)
      VizCube(t,l)

    for i in range(5):
      theta= [Rand(-math.pi,math.pi) for d in range(D)]
      sim.MoveToTheta(t,l,theta)
      sim.SimSleep(t,l,0.1)
      VizCube(t,l)
      sim.SimSleep(t,l,0.5)

  except Exception as e:
    sim.PrintException(e, ' in joint_chain1.py')

  finally:
    sim.StopPubSub(t,l)
    t.srvp.ode_pause()

  sim.Cleanup(t)

