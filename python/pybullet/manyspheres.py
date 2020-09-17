#!/usr/bin/python
#\file    manyspheres.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Sep.15, 2020
import pybullet as p
import time
import pybullet_data

if __name__=='__main__':
  p.connect(p.GUI, options="--opengl2")
  #p.connect(p.DIRECT)

  p.setAdditionalSearchPath(pybullet_data.getDataPath())
  p.setInternalSimFlags(0)
  p.resetSimulation()

  p.loadURDF("plane.urdf", useMaximalCoordinates=True)
  #p.loadURDF("tray/traybox.urdf", useMaximalCoordinates=True)
  #p.loadURDF("tray_test.urdf", useMaximalCoordinates=True)
  p.loadURDF("tray_test2.urdf", useMaximalCoordinates=True)

  t0 = time.clock()

  p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME             ,1)
  p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS               ,0)
  p.configureDebugVisualizer(p.COV_ENABLE_GUI                   ,1)
  p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER         ,0)
  p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW    ,0)
  p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW  ,0)
  p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,0)

  p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

  gravXid = p.addUserDebugParameter("gravityX", -10, 10, 0)
  gravYid = p.addUserDebugParameter("gravityY", -10, 10, 0)
  gravZid = p.addUserDebugParameter("gravityZ", -10, 10, -10)
  gravX,gravY,gravZ = 0, 0, -10
  p.setPhysicsEngineParameter(numSolverIterations=10)
  p.setPhysicsEngineParameter(contactBreakingThreshold=0.001)
  #N_SPHERE= (5,5,5)
  #N_SPHERE= (7,7,7)
  N_SPHERE= (8,8,8)
  #N_SPHERE= (10,10,5)
  #N_SPHERE= (9,9,9)
  #N_SPHERE= (10,10,10)
  for i in range(N_SPHERE[0]):
    for j in range(N_SPHERE[1]):
      for k in range(N_SPHERE[2]):
        ob = p.loadURDF("sphere_1cm.urdf", [0.02 * i, 0.02 * j, 0.2 + 0.02 * k],
                        useMaximalCoordinates=True)

  t_objects = time.clock() - t0
  t_object = t_objects / float(N_SPHERE[0]*N_SPHERE[1]*N_SPHERE[2])


  p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
  p.setGravity(0, 0, -10)

  #p.setRealTimeSimulation(1)
  #while True:
    #gravX = p.readUserDebugParameter(gravXid)
    #gravY = p.readUserDebugParameter(gravYid)
    #gravZ = p.readUserDebugParameter(gravZid)
    #p.setGravity(gravX, gravY, gravZ)
    #time.sleep(0.01)

  p.setRealTimeSimulation(0)
  p.setTimeStep(0.005)
  t0 = time.clock()
  n_steps = 300
  print 'Simulation started.'
  with open('/tmp/timestep.dat','w') as fp:
    while True:
      ts0 = time.clock()
      gravX = p.readUserDebugParameter(gravXid)
      gravY = p.readUserDebugParameter(gravYid)
      gravZ = p.readUserDebugParameter(gravZid)
      p.setGravity(gravX, gravY, gravZ)
      #time.sleep(0.01)
      ts1 = time.clock()
      p.stepSimulation()
      ts2 = time.clock()
      dts0,dts1= ts1-ts0,ts2-ts1
      #print 'Time step:',dts0,dts1
      fp.write('{0} {1} {2}\n'.format(time.clock()-t0,dts0,dts1))
  t_simulation = time.clock() - t0
  t_step = t_simulation / float(n_steps)
  total = t_simulation + t_object

  p.disconnect()
  time.sleep(2)

  print
  print "-------------------------------------------------"
  print '                 \t  Iter  \t Total  '
  print 'Object Creation  \t{0:.3e}s\t{1:.3f}s'.format(t_object, t_objects)
  print 'Simulation       \t{0:.3e}s\t{1:.3f}s'.format(t_step, t_simulation)
  print "-------------------------------------------------"
  print 'Simulation       \t        \t{0:.3f}s'.format(total)
