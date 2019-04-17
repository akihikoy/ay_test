#!/usr/bin/python
#\file    dummy_moto1.py
#\brief   Dummy motoman robot.
#         This subscribes joint_path_command and send joint_states.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Nov.10, 2017
import roslib;
roslib.load_manifest('motoman_driver')
roslib.load_manifest('sensor_msgs')
import rospy
import sensor_msgs.msg
import trajectory_msgs.msg
import threading, copy

class TRobotDummyMoto(object):
  def __init__(self):
    self.rate= 100  #/joint_states is published at 100 Hz

    self.pub_js= rospy.Publisher('/joint_states', sensor_msgs.msg.JointState, queue_size=1)
    self.sub_jpc= rospy.Subscriber('/joint_path_command', trajectory_msgs.msg.JointTrajectory, self.PathCmdCallback)

    self.js= sensor_msgs.msg.JointState()
    self.js.name= rospy.get_param('controller_joint_names')
    self.js.header.seq= 0
    self.js.header.frame_id= ''
    self.js.position= [0.0]*7
    self.js.velocity= [0.0]*7
    self.js.effort= [0.0]*7

    self.js_locker= threading.RLock()
    self.th_sendst= threading.Thread(name='SendStates', target=self.SendStates)
    self.th_sendst.start()

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
    q_traj= [p.positions for p in msg.points]
    dq_traj= [p.velocities for p in msg.points]
    t_traj= [p.time_from_start for p in msg.points]

    t_prev= rospy.Duration(0.0)
    for q,t in zip(q_traj,t_traj):
      print q,t,(t-t_prev).to_sec()
      rospy.sleep((t-t_prev).to_sec())
      t_prev= t
      with self.js_locker:
        self.js.position= copy.deepcopy(q)

if __name__=='__main__':
  rospy.init_node('dummy_motoman')
  robot= TRobotDummyMoto()
  rospy.spin()

