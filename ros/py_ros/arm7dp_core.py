#!/usr/bin/python
#\file    arm7dp_core.py
#\brief   Using ode1/arm7_door_push_node.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Oct.29, 2015
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
from geom_util import XToGPose, AngleMod1

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

#t is a dummy of core_tool.TCoreTool
def Initialize():
  t= TContainer(debug=False)
  t.pub= TContainer(debug=False)  #Publishers
  t.sub= TContainer(debug=False)  #Subscribers
  t.srvp= TContainer(debug=False)  #Service proxies
  l= TContainer(debug=False)
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
    print 'Waiting for /arm7dp_sim/get_config...'
    rospy.wait_for_service('/arm7dp_sim/get_config',3.0)
    t.srvp.ode_get_config= rospy.ServiceProxy('/arm7dp_sim/get_config', ode1.srv.ODEGetConfig2, persistent=False)
  if 'ode_reset2' not in t.srvp:
    print 'Waiting for /arm7dp_sim/reset2...'
    rospy.wait_for_service('/arm7dp_sim/reset2',3.0)
    t.srvp.ode_reset2= rospy.ServiceProxy('/arm7dp_sim/reset2', ode1.srv.ODESetConfig2, persistent=False)
  if 'ode_pause' not in t.srvp:
    print 'Waiting for /arm7dp_sim/pause...'
    rospy.wait_for_service('/arm7dp_sim/pause',3.0)
    t.srvp.ode_pause= rospy.ServiceProxy('/arm7dp_sim/pause', std_srvs.srv.Empty, persistent=False)
  if 'ode_resume' not in t.srvp:
    print 'Waiting for /arm7dp_sim/resume...'
    rospy.wait_for_service('/arm7dp_sim/resume',3.0)
    t.srvp.ode_resume= rospy.ServiceProxy('/arm7dp_sim/resume', std_srvs.srv.Empty, persistent=False)

def SetupPubSub(t,l):
  StopPubSub(t,l)
  if 'ode_control' not in t.pub:
    t.pub.ode_control= rospy.Publisher("/arm7dp_sim/control", ode1.msg.ODEControl2)
  if 'ode_viz' not in t.pub:
    t.pub.ode_viz= rospy.Publisher("/arm7dp_sim/viz", ode1.msg.ODEViz)
  if 'ode_sensors' not in t.sub:
    t.sub.ode_sensors= rospy.Subscriber("/arm7dp_sim/sensors", ode1.msg.ODESensor2, lambda msg:ODESensorCallback(msg,t,l))
  if 'ode_keyevent' not in t.sub:
    t.sub.ode_keyevent= rospy.Subscriber("/arm7dp_sim/keyevent", std_msgs.msg.Int32, lambda msg:ODEKeyEventCallback(msg,t,l))
  if 'sensor_callback' not in l:
    l.sensor_callback= None
  if 'keyevent_callback' not in l:
    l.keyevent_callback= None
  if 'control_callback' not in l:
    l.control_callback= None

def StopPubSub(t,l):
  l.sensor_callback= None
  l.keyevent_callback= None
  l.control_callback= None
  if 'ode_sensors' in t.sub:
    t.sub.ode_sensors.unregister()
    del t.sub.ode_sensors
  if 'ode_keyevent' in t.sub:
    t.sub.ode_keyevent.unregister()
    del t.sub.ode_keyevent

def ODESensorCallback(msg,t,l):
  l.sensors= msg
  if l.sensor_callback is not None:
    l.sensor_callback()

def ODEKeyEventCallback(msg,t,l):
  if l.keyevent_callback is not None:
    l.keyevent_callback(msg.data)

def GetConfig(t):
  return t.srvp.ode_get_config().config

def ResetConfig(t,config):
  t.pub.ode_viz.publish(ode1.msg.ODEViz())  #Clear visualization
  req= ode1.srv.ODESetConfig2Request()
  req.config= config
  t.srvp.ode_reset2(req)

def SimSleep(t,l,dt):
  tc0= l.sensors.time
  while l.sensors.time-tc0<dt:
    time.sleep(dt*0.02)

def MoveDTheta(t,l,dth):
  dt= l.config.TimeStep
  theta0= l.sensors.joint_angles
  theta_msg= ode1.msg.ODEControl2()
  theta_msg.angles= np.array(theta0) + dth
  t.pub.ode_control.publish(theta_msg)
  SimSleep(t,l,dt)
  if l.control_callback!=None:
    l.control_callback()

def MoveToTheta(t,l,th,dth_max=0.5):
  cnt= True
  while cnt and not rospy.is_shutdown():
    theta= l.sensors.joint_angles
    dth= np.array(map(AngleMod1,np.array(th) - theta))
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

  sim.rospy.init_node('arm7dp_test')
  t,l= sim.Initialize()

  #Visualize a box at end-link pose
  def VizCube(t,l):
    msg= sim.ode1.msg.ODEViz()
    prm= sim.ode1.msg.ODEVizPrimitive()
    prm.type= prm.CUBE
    prm.pose= sim.XToGPose(l.sensors.link_x[-7:])
    #prm.pose= sim.XToGPose(l.sensors.box1_x[-7:])
    #prm.pose= XToGPose(sim.GetCOM(l)+[0.0,0.0,0.0,1.0])
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
    l.config.JointNum= 7
    l.config.Box1Density1= 0.01
    l.config.Box1Density2= 0.5
    l.config.FixedBase= True
    l.config.ObjectMode= 1  #0: None, 1: Box1, ...

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
    sim.PrintException(e, ' in arm7dp_core.py')

  finally:
    sim.StopPubSub(t,l)
    t.srvp.ode_pause()

  sim.Cleanup(t)

