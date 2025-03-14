#!/usr/bin/python3
#\file    dummy_robot.py
#\brief   Dummy robot ver.2.
#         This subscribes joint_path_command topic, follow_joint_trajectory (actionlib) topic,
#         and joint_speed_command,
#         This publishes joint_states.
#         The trajectory is interpolated with a spline.
#         This is useful to simulate kinematic motion.
#         This can simulate Motoman, Mikata arm, and UR.
#         Usage:
#           Motoman: ./dummy_robot2.py 100
#           UR:      ./dummy_robot2.py 125 joint_speed_command:=/ur_driver/joint_speed
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Nov.10, 2017
#\version 0.2
#\date    Jun.21, 2018
import roslib;
roslib.load_manifest('sensor_msgs')
roslib.load_manifest('ay_py')
import rospy
import sensor_msgs.msg
import trajectory_msgs.msg
import actionlib
import control_msgs.msg
import threading, copy, sys
from ay_py.core import TCubicHermiteSpline

class TDummyRobot(object):
  def __init__(self, rate=100):
    self.rate= rate  #/joint_states is published at rate Hz

    self.pub_js= rospy.Publisher('/joint_states', sensor_msgs.msg.JointState, queue_size=1)
    self.sub_jpc= rospy.Subscriber('/joint_path_command', trajectory_msgs.msg.JointTrajectory, self.PathCmdCallback)
    self.sub_jsc= rospy.Subscriber('/joint_speed_command', trajectory_msgs.msg.JointTrajectory, self.SpeedCmdCallback, queue_size=1)

    self.js= sensor_msgs.msg.JointState()
    self.js.name= rospy.get_param('controller_joint_names')
    self.dof= len(self.js.name)
    self.js.header.seq= 0
    self.js.header.frame_id= ''
    self.js.position= [0.0]*self.dof
    self.js.velocity= [0.0]*self.dof
    self.js.effort= [0.0]*self.dof

    self.follow_traj_active= False

    self.js_locker= threading.RLock()
    self.th_sendst= threading.Thread(name='SendStates', target=self.SendStates)
    self.th_sendst.start()

    self.th_follow_traj= None

    self.ftaction_feedback= control_msgs.msg.FollowJointTrajectoryFeedback()
    self.ftaction_result= control_msgs.msg.FollowJointTrajectoryResult()
    self.ftaction_name= '/follow_joint_trajectory'
    self.ftaction_actsrv= actionlib.SimpleActionServer(self.ftaction_name, control_msgs.msg.FollowJointTrajectoryAction, execute_cb=self.FollowTrajActionCallback, auto_start=False)
    self.ftaction_actsrv.start()

  def SendStates(self):
    rate= rospy.Rate(self.rate)
    while not rospy.is_shutdown():
      with self.js_locker:
        self.js.header.seq= self.js.header.seq+1
        self.js.header.stamp= rospy.Time.now()
        #print self.js.position
        self.pub_js.publish(self.js)
      rate.sleep()

  def PathCmdCallback(self, msg):
    if self.follow_traj_active:
      self.follow_traj_active= False
      self.th_follow_traj.join()
    self.follow_traj_active= True
    self.th_follow_traj= threading.Thread(name='FollowTraj', target=lambda:self.FollowTraj(msg))
    self.th_follow_traj.start()

  def SpeedCmdCallback(self, msg):
    if self.follow_traj_active:
      return
    self.follow_traj_active= True
    rate= rospy.Rate(self.rate)
    dt= 1.0/self.rate
    q= [q0+v*dt for q0,v in zip(self.js.position,msg.points[0].velocities)]
    with self.js_locker:
      self.js.position= q
      self.js.velocity= msg.points[0].velocities
    rate.sleep()
    self.follow_traj_active= False

  def FollowTraj(self, traj):
    splines,T= self.GetSplines(traj)
    rate= rospy.Rate(self.rate)
    t0= rospy.Time.now()
    while all(((rospy.Time.now()-t0)<T, self.follow_traj_active, not rospy.is_shutdown())):
      t= (rospy.Time.now()-t0).to_sec()
      q_dq= [splines[d].Evaluate(t,with_tan=True) for d in range(self.dof)]
      q= [q for q,dq in q_dq]
      dq= [dq for q,dq in q_dq]
      #print t, q
      with self.js_locker:
        self.js.position= q
        self.js.velocity= dq
      rate.sleep()
    self.follow_traj_active= False

  def GetSplines(self, traj):
    q_traj= [p.positions for p in traj.points]
    dq_traj= [p.velocities for p in traj.points]
    t_traj= [p.time_from_start for p in traj.points]
    #If no initial point:
    if t_traj[0].to_sec()>1.0e-3:
      q_traj= [self.js.position]+q_traj
      dq_traj= [self.js.velocity]+dq_traj
      t_traj= [rospy.Duration(0.0)]+t_traj
    print('Received trajectory command:')
    print([t.to_sec() for t in t_traj])
    print(q_traj)

    #Modeling the trajectory with spline.
    splines= [TCubicHermiteSpline() for d in range(self.dof)]
    for d in range(len(splines)):
      data_d= [[t.to_sec(),q[d]] for q,t in zip(q_traj,t_traj)]
      splines[d].Initialize(data_d, tan_method=splines[d].CARDINAL, c=0.0, m=0.0)

    return splines, t_traj[-1]

  def FollowTrajActionCallback(self, goal):
    if self.follow_traj_active:
      return
    self.follow_traj_active= True

    splines,T= self.GetSplines(goal.trajectory)
    success= True
    rate= rospy.Rate(self.rate)
    t0= rospy.Time.now()
    while all(((rospy.Time.now()-t0)<T, self.follow_traj_active, not rospy.is_shutdown())):
      if self.ftaction_actsrv.is_preempt_requested():
        print('%s: Preempted' % self.ftaction_name)
        self.ftaction_actsrv.set_preempted()
        success= False
        break

      t= (rospy.Time.now()-t0).to_sec()
      q_dq= [splines[d].Evaluate(t,with_tan=True) for d in range(self.dof)]
      q= [q for q,dq in q_dq]
      dq= [dq for q,dq in q_dq]
      #print t, q
      with self.js_locker:
        self.js.position= q
        self.js.velocity= dq

      self.ftaction_actsrv.publish_feedback(self.ftaction_feedback)
      rate.sleep()

    if self.follow_traj_active and success:
      self.ftaction_result.error_code= self.ftaction_result.SUCCESSFUL
      print('%s: Succeeded' % self.ftaction_name)
      self.ftaction_actsrv.set_succeeded(self.ftaction_result)
    self.follow_traj_active= False


if __name__=='__main__':
  rospy.init_node('dummy_robot')
  rate= float(sys.argv[1]) if len(sys.argv)>1 else 100.0
  robot= TDummyRobot(rate)
  rospy.spin()
