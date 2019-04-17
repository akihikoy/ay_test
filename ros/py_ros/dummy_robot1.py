#!/usr/bin/python
#\file    dummy_robot.py
#\brief   Dummy robot.
#         This subscribes joint_path_command and send joint_states.
#         The trajectory is interpolated with a spline.
#         This is useful to simulate kinematic motion.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Nov.10, 2017
import roslib;
#roslib.load_manifest('motoman_driver')
roslib.load_manifest('sensor_msgs')
roslib.load_manifest('ay_py')
import rospy
import sensor_msgs.msg
import trajectory_msgs.msg
import threading, copy, sys
from ay_py.core import TCubicHermiteSpline

class TDummyRobot(object):
  def __init__(self, rate=100):
    self.rate= rate  #/joint_states is published at rate Hz

    self.pub_js= rospy.Publisher('/joint_states', sensor_msgs.msg.JointState, queue_size=1)
    self.sub_jpc= rospy.Subscriber('/joint_path_command', trajectory_msgs.msg.JointTrajectory, self.PathCmdCallback)

    self.js= sensor_msgs.msg.JointState()
    self.js.name= rospy.get_param('controller_joint_names')
    self.dof= len(self.js.name)
    self.js.header.seq= 0
    self.js.header.frame_id= ''
    self.js.position= [0.0]*self.dof
    self.js.velocity= [0.0]*self.dof
    self.js.effort= [0.0]*self.dof

    self.js_locker= threading.RLock()
    self.th_sendst= threading.Thread(name='SendStates', target=self.SendStates)
    self.th_sendst.start()

    self.th_follow_traj= None
    self.follow_traj_active= False

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

  def FollowTraj(self, traj):
    q_traj= [p.positions for p in traj.points]
    dq_traj= [p.velocities for p in traj.points]
    t_traj= [p.time_from_start for p in traj.points]
    #If no initial point:
    if t_traj[0].to_sec()>1.0e-3:
      q_traj= [self.js.position]+q_traj
      dq_traj= [self.js.velocity]+dq_traj
      t_traj= [rospy.Duration(0.0)]+t_traj
    print 'Received trajectory command:'
    print [t.to_sec() for t in t_traj]
    print q_traj

    #Modeling the trajectory with spline.
    splines= [TCubicHermiteSpline() for d in range(self.dof)]
    for d in range(len(splines)):
      data_d= [[t.to_sec(),q[d]] for q,t in zip(q_traj,t_traj)]
      splines[d].Initialize(data_d, tan_method=splines[d].CARDINAL, c=0.0, m=0.0)

    rate= rospy.Rate(self.rate)
    t0= rospy.Time.now()
    while all(((rospy.Time.now()-t0)<t_traj[-1], self.follow_traj_active, not rospy.is_shutdown())):
      t= (rospy.Time.now()-t0).to_sec()
      q= [splines[d].Evaluate(t) for d in range(self.dof)]
      #print t, q
      with self.js_locker:
        self.js.position= copy.deepcopy(q)
      rate.sleep()

if __name__=='__main__':
  rospy.init_node('dummy_robot')
  rate= float(sys.argv[1]) if len(sys.argv)>1 else 100.0
  robot= TDummyRobot(rate)
  rospy.spin()

