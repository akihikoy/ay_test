#!/usr/bin/python
#\file    bxtr.py
#\brief   Baxter utility class
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Oct.29, 2015
'''
NOTE: run beforehand:
  $ rosrun baxter_interface joint_trajectory_action_server.py
'''

import roslib
import rospy
import actionlib
import std_msgs.msg
import sensor_msgs.msg
import control_msgs.msg
import trajectory_msgs.msg
import baxter_interface
import time, math, sys, copy
import cv2
import cv_bridge

RIGHT=0
LEFT=1
def LRTostr(whicharm):
  if whicharm==RIGHT: return 'right'
  if whicharm==LEFT:  return 'left'
  return None

'''Baxter utility class'''
class TRobotBaxter:
  def __init__(self):
    pass

  '''Initialize (e.g. establish ROS connection).'''
  def Init(self):
    self.limbs= [None,None]
    self.limbs[RIGHT]= baxter_interface.Limb(LRTostr(RIGHT))
    self.limbs[LEFT]=  baxter_interface.Limb(LRTostr(LEFT))

    self.joint_names= [[],[]]
    #self.joint_names[RIGHT]= ['right_'+joint for joint in ['s0', 's1', 'e0', 'e1', 'w0', 'w1', 'w2']]
    #self.joint_names[LEFT]=  ['left_' +joint for joint in ['s0', 's1', 'e0', 'e1', 'w0', 'w1', 'w2']]
    self.joint_names[RIGHT]= self.limbs[RIGHT].joint_names()
    self.joint_names[LEFT]=  self.limbs[LEFT].joint_names()

    self.head= baxter_interface.Head()

    self.client= [None,None]
    self.client[RIGHT]= actionlib.SimpleActionClient('/robot/limb/%s/follow_joint_trajectory'%LRTostr(RIGHT), control_msgs.msg.FollowJointTrajectoryAction)
    self.client[LEFT]= actionlib.SimpleActionClient('/robot/limb/%s/follow_joint_trajectory'%LRTostr(LEFT), control_msgs.msg.FollowJointTrajectoryAction)
    def WaitClient(c):
      if not c.wait_for_server(rospy.Duration(5.0)):
        rospy.logerr('Exiting - Joint Trajectory Action Server Not Found')
        rospy.logerr('Run: rosrun baxter_interface joint_trajectory_action_server.py')
        rospy.signal_shutdown('Action Server not found')
        sys.exit(1)
    WaitClient(self.client[RIGHT])
    WaitClient(self.client[LEFT])

    self.default_face= 'baymax.jpg'
    self.ChangeFace(self.default_face)

    self.bxjs= BxJointSprings()  #virtual joint spring controller

  def Cleanup(self):
    #NOTE: cleaning-up order is important. consider dependency
    pass

  @property
  def BaseFrame(self):
    return 'torso'

  '''End link of an arm.'''
  def EndLink(self, arm):
    if   arm==RIGHT:  return 'right_gripper'
    elif arm==LEFT:   return 'left_gripper'

  '''Names of joints of an arm.'''
  def JointNames(self, arm):
    return self.joint_names[arm]

  '''Return joint angles of an arm.
    arm: LEFT, RIGHT, or None (==currarm). '''
  def Q(self, arm=None):
    if arm==None:  arm= self.Arm
    #with self.sensor_locker:  #TODO:FIXME
    angles= self.limbs[arm].joint_angles()  #FIXME:does this include deepcopy?
    q= [angles[joint] for joint in self.joint_names[arm]]  #Serialize
    return q

  '''Follow a joint angle trajectory.
    arm: LEFT, RIGHT, or None (==currarm).
    q_traj: joint angle trajectory [q0,...,qD]*N.
    t_traj: corresponding times in seconds from start [t1,t2,...,tN].
    blocking: False: move background, True: wait until motion ends, 'time': wait until tN. '''
  def FollowQTraj(self, q_traj, t_traj, arm=None, blocking=False):
    assert(len(q_traj)==len(t_traj))
    if arm==None:  arm= self.Arm

    #copy q_traj, t_traj to goal
    goal= control_msgs.msg.FollowJointTrajectoryGoal()
    goal.goal_time_tolerance= rospy.Time(0.1)
    goal.trajectory.joint_names= self.joint_names[arm]
    #goal.trajectory= ToROSTrajectory(self.JointNames(arm), q_traj, t_traj)  #TODO:FIXME:use this!
    for ts,q in zip(t_traj,q_traj):
      point= trajectory_msgs.msg.JointTrajectoryPoint()
      point.positions= copy.deepcopy(q)
      point.time_from_start= rospy.Duration(ts)
      goal.trajectory.points.append(point)
    goal.trajectory.header.stamp= rospy.Time.now()

    #with self.control_locker:  #TODO:FIXME
    actc= self.client[RIGHT] if arm==RIGHT else self.client[LEFT]
    actc.send_goal(goal)
    #BlockAction(actc, blocking=blocking, duration=t_traj[-1])  #TODO:FIXME:use this!
    if blocking:  actc.wait_for_result(timeout=rospy.Duration(t_traj[-1]+5.0))
    #actc.get_result()
    return actc

  def ChangeFace(self, file_name):
    img= cv2.imread(file_name)
    msg= cv_bridge.CvBridge().cv2_to_imgmsg(img, encoding="bgr8")
    pub= rospy.Publisher('/robot/xdisplay', sensor_msgs.msg.Image, latch=True, queue_size=1)
    pub.publish(msg)
    # Sleep to allow for image to be published.
    #rospy.sleep(1)

  def MoveHeadPan(self, angle, speed=100, timeout=10):
    self.head.set_pan(angle, speed, timeout)  #NOTE: Default speed=100, timeout=10

  def NodHead(self):
    self.head.command_nod()

  def ActivateJointSprings(self, arms=(RIGHT,LEFT), target_angles=(None,None), stop_err=None, stop_dt=None):
    self.bxjs.AttachSprings(arms=arms, target_angles=target_angles, stop_err=stop_err, stop_dt=stop_dt)


#Utility

def EnableBaxter():
  rs= baxter_interface.RobotEnable(baxter_interface.CHECK_VERSION)
  init_state= rs.state().enabled
  def clean_shutdown():
    if not init_state:
      print 'Disabling robot...'
      rs.disable()
  rospy.on_shutdown(clean_shutdown)
  rs.enable()

def DecomposePoseTraj(poses):
  t_traj= [ts for ts,q_r,q_l in poses]
  qr_traj= [q_r for ts,q_r,q_l in poses]
  ql_traj= [q_l for ts,q_r,q_l in poses]
  return t_traj,qr_traj,ql_traj


import dynamic_reconfigure.server
import baxter_examples.cfg.JointSpringsExampleConfig

#src: /opt/ros/hydro/lib/baxter_examples/joint_torque_springs.py
'''
Virtual Joint Springs class for torque example.

@param limb: limb on which to run joint springs example
@param reconfig_server: dynamic reconfigure server

JointSprings class contains methods for the joint torque example allowing
moving the limb to a neutral location, entering torque mode, and attaching
virtual springs.
'''
class BxJointSprings(object):
  def __init__(self, reconfig_server=None):
    if reconfig_server==None:
      reconfig_server= dynamic_reconfigure.server.Server(baxter_examples.cfg.JointSpringsExampleConfig,
                              lambda config, level: config)
    self.dyn= reconfig_server

    self.arms= None

    # control parameters
    self.rate= 1000.0  # Hz
    self.missed_cmds= 20.0  # Missed cycles before triggering timeout

    #springs= [10.0, 15.0, 5.0, 5.0, 3.0, 2.0, 1.5]  #ORIGINAL
    #springs= [10.0, 15.0, 5.0, 10.0, 3.0, 8.0, 1.5]
    #springs= [10.0, 15.0, 10.0, 15.0, 3.0, 8.0, 1.5]
    springs= [20.0, 20.0, 20.0, 20.0, 10.0, 15.0, 5.0]
    damping= [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

    # create our limb instance
    self.limbs= [None,None]
    self.limbs[RIGHT]= baxter_interface.Limb(LRTostr(RIGHT))
    self.limbs[LEFT]=  baxter_interface.Limb(LRTostr(LEFT))

    self.joint_names= [[],[]]
    self.joint_names[RIGHT]= self.limbs[RIGHT].joint_names()
    self.joint_names[LEFT]=  self.limbs[LEFT].joint_names()

    # initialize parameters
    self.springs= [{joint:springs[j] for j,joint in enumerate(self.joint_names[RIGHT])},
                   {joint:springs[j] for j,joint in enumerate(self.joint_names[LEFT])}]
    self.damping= [{joint:damping[j] for j,joint in enumerate(self.joint_names[RIGHT])},
                   {joint:damping[j] for j,joint in enumerate(self.joint_names[LEFT])}]
    self.start_angles= [dict(),dict()]

    # create cuff disable publisher
    self.pub_cuff_disable= [None,None]
    self.pub_cuff_disable[RIGHT]= rospy.Publisher('/robot/limb/%s/suppress_cuff_interaction'%LRTostr(RIGHT), std_msgs.msg.Empty, queue_size=1)
    self.pub_cuff_disable[LEFT]= rospy.Publisher('/robot/limb/%s/suppress_cuff_interaction'%LRTostr(LEFT), std_msgs.msg.Empty, queue_size=1)

  def UpdateParameters(self):
    for arm in self.arms:
      for joint in self.joint_names[arm]:
        self.springs[arm][joint]= self.dyn.config[joint[-2:]+'_spring_stiffness']
        self.damping[arm][joint]= self.dyn.config[joint[-2:]+'_damping_coefficient']

  def UpdateForces(self):
    """
    Calculates the current angular difference between the start position
    and the current joint positions applying the joint torque spring forces
    as defined on the dynamic reconfigure server.
    """
    # get latest spring constants
    #self.UpdateParameters()

    self.angle_errs= [dict(),dict()]
    for arm in self.arms:
      # disable cuff interaction
      self.pub_cuff_disable[arm].publish()

      # create our command dict
      cmd= dict()
      # record current angles/velocities
      cur_pos= self.limbs[arm].joint_angles()
      cur_vel= self.limbs[arm].joint_velocities()
      # calculate current forces
      for joint in self.joint_names[arm]:
        # spring portion
        self.angle_errs[arm][joint]= self.start_angles[arm][joint]-cur_pos[joint]
        cmd[joint]= self.springs[arm][joint]*self.angle_errs[arm][joint]
        # damping portion
        cmd[joint]-= self.damping[arm][joint]*cur_vel[joint]
      # command new joint torques
      self.limbs[arm].set_joint_torques(cmd)
      #if arm==RIGHT: print (self.angle_errs[RIGHT]['right_w1'],cmd['right_w1']),
      #else:          print (self.angle_errs[LEFT]['left_w1'],cmd['left_w1'])

  def AttachSprings(self, arms=(RIGHT,LEFT), target_angles=(None,None), stop_err=None, stop_dt=None):
    """
    Switches to joint torque mode and attached joint springs to current
    joint positions.
    """
    self.arms= arms

    # set control rate
    control_rate= rospy.Rate(self.rate)

    # record initial joint angles
    for arm in self.arms:
      if target_angles[arm] is None:
        self.start_angles[arm]= self.limbs[arm].joint_angles()
      else:
        self.start_angles[arm]= {joint:target_angles[arm][j] for j,joint in enumerate(self.joint_names[arm])}  #Deserialize

      # for safety purposes, set the control rate command timeout.
      # if the specified number of command cycles are missed, the robot
      # will timeout and disable
      self.limbs[arm].set_command_timeout((1.0 / self.rate) * self.missed_cmds)

    # loop at specified rate commanding new joint torques
    rs= baxter_interface.RobotEnable(baxter_interface.CHECK_VERSION)
    start_time= rospy.Time.now()
    while not rospy.is_shutdown():
      if not rs.state().enabled:
        rospy.logerr('Joint torque example failed to meet specified control rate timeout.')
        break
      self.UpdateForces()
      if stop_err is not None:
        errs= []
        for arm in self.arms:
          errs.append(max(map(abs,self.angle_errs[arm].values()) if self.angle_errs[arm] is not None else 0.0))
        max_err= max(errs)
        #print max_err
        if max_err>stop_err:  break
      control_rate.sleep()
      if stop_dt is not None:
        dt= (rospy.Time.now() - start_time).to_sec()
        if dt>stop_dt:  break
    self.CleanShutdown()
    #print 'self.springs=',[self.springs[RIGHT][joint] for joint in self.joint_names[RIGHT]],
    #print                 [self.springs[LEFT][joint] for joint in self.joint_names[LEFT]]
    #print 'self.damping=',[self.damping[RIGHT][joint] for joint in self.joint_names[RIGHT]],
    #print                 [self.damping[LEFT][joint] for joint in self.joint_names[LEFT]]

  def CleanShutdown(self):
    """
    Switches out of joint torque mode to exit cleanly
    """
    for arm in self.arms:
      print 'Exiting virtual spring control mode',LRTostr(arm)
      self.limbs[arm].exit_control_mode()
      print 'done'
      #self.limbs[arm].move_to_joint_positions(self.limbs[arm].joint_angles(), timeout=0.0)


