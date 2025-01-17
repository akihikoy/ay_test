#!/usr/bin/python3
#Simplified flow dynamics model
import threading
import time
import math
import random

class TFlowDyn:
  def __init__(self,a_bottle=1.0,a_cup=0.0,time_step=0.05,show_state=True):
    self.a_bottle= a_bottle
    self.a_cup= a_cup
    self.theta= 0.0
    self.time_step= time_step
    self.show_state= show_state

    self.flow= 0.0
    self.flow_obs= 0.0
    self.dtheta= 0.0
    self.time= 0.0
    self.t_start= 0.0
    self.running= False

  def __del__(self):
    self.Stop()

  def Step(self):
    theta_fs= -0.5*math.pi*self.a_bottle + 0.6666*math.pi
    if self.theta>theta_fs:
      self.flow= min(4.0*(self.theta-theta_fs), self.a_bottle/self.time_step)
    else:
      self.flow= 0.0
    self.flow_obs= self.flow + 0.01*(random.random()-0.5)
    self.a_bottle+= self.time_step * (-self.flow)
    self.a_cup+= self.time_step * self.flow
    self.theta+= self.time_step * self.dtheta
    if self.theta<0.0:  self.theta= 0.0
    elif self.theta>math.pi:  self.theta= math.pi
    self.time= time.time()-self.t_start

    if self.show_state:
      print('%f %f %f %f %f %f %f' % (self.time, self.a_bottle, self.a_cup, self.flow, self.flow_obs, self.theta, self.dtheta))
    self.fp.write('%f %f %f %f %f %f %f\n' % (self.time, self.a_bottle, self.a_cup, self.flow, self.flow_obs, self.theta, self.dtheta))
    time.sleep(self.time_step)

  def Loop(self):
    while self.running:
      self.Step()

  def Start(self):
    self.running= True
    self.fp= open('data/flow.dat','w')
    self.t_start= time.time()
    self.tl= threading.Thread(name='loop', target=self.Loop)
    self.tl.start()

  def Stop(self):
    if self.running:
      self.running= False
      self.tl.join()
      self.fp.close()

  def Control(self,dtheta):
    self.dtheta= dtheta

if __name__=='__main__':
  fdyn= TFlowDyn()
  fdyn.Start()
  time.sleep(1.0)
  fdyn.Control(1.0)
  time.sleep(2.0)
  fdyn.Control(0.0)
  time.sleep(3.0)
  fdyn.Stop()
  print('Plot by:')
  print("cat data/flow.dat | qplot -x -s 'set y2tics' - u 1:3 - u 1:4 - u 1:5 - u 1:6 ax x1y2")

