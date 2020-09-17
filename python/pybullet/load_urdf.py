#!/usr/bin/python
#\file    load_urdf.py
#\brief   Load URDF.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Sep.17, 2020
import pybullet as pb
import time
import pybullet_data
import os,random

def LoadXacro(fileName, basePosition=[0.,0.,0.], baseOrientation=[0.,0.,0.,1.], useMaximalCoordinates=0, useFixedBase=0, flags=0, globalScaling=1.0, physicsClientId=0, args='', tmpFileFmt='/tmp/pybullet{code}.urdf'):
  while True:
    tmpFileName= tmpFileFmt.format(code=random.randint(10000000,99999999))
    if not os.path.exists(tmpFileName):  break
  print 'Converting Xacro {inFile} to URDF {outFile}'.format(inFile=fileName,outFile=tmpFileName)
  os.system('rosrun xacro xacro --inorder {inFile} -o {outFile} {args}'.format(inFile=fileName,outFile=tmpFileName,args=args))
  pb.loadURDF(tmpFileName, basePosition=basePosition, baseOrientation=baseOrientation,
              useMaximalCoordinates=useMaximalCoordinates, useFixedBase=useFixedBase,
              flags=flags, globalScaling=globalScaling, physicsClientId=physicsClientId)

if __name__=='__main__':
  pb.connect(pb.GUI, options="--opengl2")
  #pb.connect(pb.DIRECT)

  pb.setAdditionalSearchPath(pybullet_data.getDataPath())
  pb.setInternalSimFlags(0)
  pb.resetSimulation()

  pb.loadURDF("plane.urdf", useMaximalCoordinates=True)
  #pb.loadURDF("tray/traybox.urdf", useMaximalCoordinates=True)
  #pb.loadURDF("tray_test.urdf", useMaximalCoordinates=True)
  #pb.loadURDF("tray_test2.urdf", useMaximalCoordinates=True)
  #pb.loadURDF("container1.urdf", useMaximalCoordinates=True)
  #pb.loadURDF("container2.urdf", useMaximalCoordinates=True)
  LoadXacro("data/container1.urdf.xacro", useMaximalCoordinates=True, args='Size2H:=0.03 SizeXY:=0.4')

  #pb.configureDebugVisualizer(pb.COV_ENABLE_WIREFRAME             ,1)
  pb.configureDebugVisualizer(pb.COV_ENABLE_SHADOWS               ,0)
  pb.configureDebugVisualizer(pb.COV_ENABLE_GUI                   ,1)
  pb.configureDebugVisualizer(pb.COV_ENABLE_TINY_RENDERER         ,0)
  pb.configureDebugVisualizer(pb.COV_ENABLE_RGB_BUFFER_PREVIEW    ,0)
  pb.configureDebugVisualizer(pb.COV_ENABLE_DEPTH_BUFFER_PREVIEW  ,0)
  pb.configureDebugVisualizer(pb.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,0)

  pb.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 0)

  gravXid = pb.addUserDebugParameter("gravityX", -10, 10, 0)
  gravYid = pb.addUserDebugParameter("gravityY", -10, 10, 0)
  gravZid = pb.addUserDebugParameter("gravityZ", -10, 10, -10)
  gravX,gravY,gravZ = 0, 0, -10
  pb.setPhysicsEngineParameter(numSolverIterations=10)
  pb.setPhysicsEngineParameter(contactBreakingThreshold=0.001)

  pb.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 1)
  pb.setGravity(0, 0, -10)

  pb.setRealTimeSimulation(1)
  while True:
    gravX = pb.readUserDebugParameter(gravXid)
    gravY = pb.readUserDebugParameter(gravYid)
    gravZ = pb.readUserDebugParameter(gravZid)
    pb.setGravity(gravX, gravY, gravZ)
    time.sleep(0.01)

  pb.disconnect()
