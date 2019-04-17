#!/usr/bin/python
#\file    fk_test.py
#\brief   Test learned forward kinematics.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.05, 2015

import joint_chain1 as sim
import time,sys
import math,random

def Rand(xmin=-0.5,xmax=0.5):
  return random.random()*(xmax-xmin)+xmin

#Visualize a box at end-link pose
def AddVizCube(msg,x,c=(0.4,1.0,0.4, 0.5),p=(0.15, 0.12, 0.06)):
  #msg= sim.ode1.msg.ODEViz()
  prm= sim.ode1.msg.ODEVizPrimitive()
  prm.type= prm.CUBE
  prm.pose= sim.XToGPose(x)
  prm.param= p
  prm.color= sim.RGBA(*c)
  msg.objects.append(prm)
  #t.pub.ode_viz.publish(msg)

class TFKTester:
  def __init__(self, dof, dth_max=0.2):
    sim.rospy.init_node('ros_min')
    self.t,self.l= sim.Initialize()
    t=self.t; l=self.l
    l.dth_max= dth_max

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

  def __del__(self):
    self.Cleanup()

  def Cleanup(self):
    if not 't' in self.__dict__ or not 'l' in self.__dict__:  return
    t=self.t; l=self.l
    sim.StopPubSub(t,l)
    t.srvp.ode_pause()
    sim.Cleanup(t)
    del self.t
    del self.l

  def Callback(self):
    t=self.t; l=self.l
    vmsg= sim.ode1.msg.ODEViz()
    AddVizCube(vmsg,l.sensors.link_x[-7:])
    x= l.f_fwdkin(l.sensors.joint_angles)
    AddVizCube(vmsg,x,c=(1.0,0.0,0.5, 0.8))
    t.pub.ode_viz.publish(vmsg)
    l.c_samples-= 1

  #Test f_fwdkin which is a function q(joint angles) --> x(pose)
  def Test(self, f_fwdkin, n_samples=100):
    t=self.t; l=self.l
    l.c_samples= n_samples
    l.f_fwdkin= f_fwdkin
    #l.sensor_callback= self.Callback
    l.control_callback= self.Callback

    t.srvp.ode_resume()
    while l.c_samples>0 and not sim.rospy.is_shutdown():
      theta= [Rand(-math.pi,math.pi) for d in range(l.config.JointNum)]
      sim.MoveToTheta(t,l,theta,dth_max=l.dth_max)
    t.srvp.ode_pause()

    l.sensor_callback= None
    l.control_callback= None
    l.f_fwdkin= None


def Test():
  tester= TFKTester(3)
  tester.Test(f_fwdkin=lambda q:[0,0,1, 0,0,0,1])
  tester.Cleanup()


def Main():
  import argparse
  import re
  import numpy as np
  import numpy.linalg as la
  from chainer import cuda, Variable
  import chainer.functions  as F
  import six.moves.cPickle as pickle
  from fk_learn import ForwardModel, ModelCodesWithXAll

  parser = argparse.ArgumentParser(description='Chainer example: regression')
  parser.add_argument('--gpu', '-g', default=-1, type=int,
                      help='GPU ID (negative value indicates CPU)')
  parser.add_argument('--dof', '-D', default='3', type=str,
                      help='DoF code')
  parser.add_argument('--mdof', '-MD', default='', type=str,
                      help='DoF code of model file. Blank uses the same one as --dof.')
  args = parser.parse_args()

  #dof = 3
  dofc= args.dof
  mdofc= args.mdof if args.mdof!='' else dofc
  dof= int(re.search('^[0-9]+',dofc).group())

  # Load model from file
  def load_model():
    global model
    model = pickle.load(open('result/fknn%s.dat'%mdofc, 'rb'))
    print 'Loaded model from:','result/fknn%s.dat'%mdofc
    if args.gpu >= 0:
      cuda.init(args.gpu)
      model.to_gpu()
  load_model()

  Dy= 7 if dofc not in ModelCodesWithXAll else 7*dof

  # Setup tester
  tester= TFKTester(dof)

  # Neural net architecture
  def forward(x_data, y_data, train=True):
    return ForwardModel(dofc, model, x_data, y_data, train)

  # Predict for a single query x
  def predict(x):
    x_batch = np.array([x]).astype(np.float32)
    y_batch = np.array([[0.0]*Dy]).astype(np.float32)  #Dummy
    if args.gpu >= 0:
      x_batch = cuda.to_gpu(x_batch)
      y_batch = cuda.to_gpu(y_batch)
    loss, pred = forward(x_batch, y_batch, train=False)
    y= cuda.to_cpu(pred.data)[0]
    #y[-4:]= [0,0,0,1]  #Modify the orientation
    y[-4:]/= la.norm(y[-4:])  #Normalize the quaternion
    return y

  for i in range(10):
    load_model()
    print predict(np.array([0.0]*dof).astype(np.float32))
    tester.Test(f_fwdkin=predict, n_samples=100)

  tester.Cleanup()


def PlotGraphs():
  print 'Plotting graphs..'
  import os
  commands=[
    '''qplot -x2 aaa 'sin(x)' w l &''',
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
