#!/usr/bin/python
#\file    pour1.py
#\brief   Pouring simulation.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Sep.17, 2020
import pybullet as pb
import time
import pybullet_data
import os,random
from load_urdf import LoadXacro

def GenerateParticles(N=200):
  BallRad= 0.01
  SrcSizeXY= 0.10
  BoxThickness= 0.005

  rad= BallRad*1.05  #margin
  xymin= -0.5*SrcSizeXY + BoxThickness+rad
  xymax= 0.5*SrcSizeXY - BoxThickness-rad
  x,y,z= xymin, xymin, BoxThickness+rad
  pos= []
  for _ in range(N):
    pos.append((x,y,z))
    x+= 2.0*rad
    if x>xymax:
      x= xymin+(x-xymax)
      y+= 2.0*rad
      if y>xymax:
        y= xymin+(y-xymax)
        z+= 2.0*rad
  for p in pos:
    ob= pb.loadURDF("sphere_1cm.urdf", basePosition=p, useMaximalCoordinates=True)

if __name__=='__main__':
  pb.connect(pb.GUI, options="--opengl2")
  #pb.connect(pb.DIRECT)

  pb.setAdditionalSearchPath(pybullet_data.getDataPath())
  pb.setInternalSimFlags(0)
  pb.resetSimulation()

  #pb.configureDebugVisualizer(pb.COV_ENABLE_WIREFRAME             ,1)
  pb.configureDebugVisualizer(pb.COV_ENABLE_SHADOWS               ,0)
  pb.configureDebugVisualizer(pb.COV_ENABLE_GUI                   ,1)
  pb.configureDebugVisualizer(pb.COV_ENABLE_TINY_RENDERER         ,0)
  pb.configureDebugVisualizer(pb.COV_ENABLE_RGB_BUFFER_PREVIEW    ,0)
  pb.configureDebugVisualizer(pb.COV_ENABLE_DEPTH_BUFFER_PREVIEW  ,0)
  pb.configureDebugVisualizer(pb.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,0)

  pb.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 0)

  pb.loadURDF("plane.urdf", useMaximalCoordinates=True)
  #pb.loadURDF("tray/traybox.urdf", useMaximalCoordinates=True)
  #pb.loadURDF("tray_test.urdf", useMaximalCoordinates=True)
  #pb.loadURDF("tray_test2.urdf", useMaximalCoordinates=True)
  #pb.loadURDF("container1.urdf", useMaximalCoordinates=True)
  #pb.loadURDF("container2.urdf", useMaximalCoordinates=True)
  LoadXacro("data/container1.urdf.xacro", useMaximalCoordinates=True,
            args='SizeXY:=0.10 SizeZ:=0.15 Size2H:=0.012')

  LoadXacro("data/container2.urdf.xacro", basePosition=[0,0.15,0], useMaximalCoordinates=True,
            args='SizeX:=0.10 SizeY:=0.12 SizeZ:=0.12')

  gravX,gravY,gravZ= 0, 0, -1
  gravXid= pb.addUserDebugParameter('gravityX', -10, 10, gravX)
  gravYid= pb.addUserDebugParameter('gravityY', -10, 10, gravY)
  gravZid= pb.addUserDebugParameter('gravityZ', -10, 10, gravZ)
  pb.setPhysicsEngineParameter(numSolverIterations=10)
  pb.setPhysicsEngineParameter(contactBreakingThreshold=0.001)

  GenerateParticles(N=400)

  pb.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 1)
  pb.setGravity(gravX,gravY,gravZ)

  pb.setRealTimeSimulation(1)
  while True:
    gravX= pb.readUserDebugParameter(gravXid)
    gravY= pb.readUserDebugParameter(gravYid)
    gravZ= pb.readUserDebugParameter(gravZid)
    pb.setGravity(gravX, gravY, gravZ)
    time.sleep(0.01)

  pb.disconnect()
