#!/usr/bin/python
#\file    follow_q_traj_change1.py
#\brief   Follow a trajectory and change the trajectory when indicated by the operator.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Oct.26, 2023
import roslib; roslib.load_manifest('motoman_driver')
import rospy
import actionlib
import control_msgs.msg
import trajectory_msgs.msg
import actionlib_msgs.msg
import sensor_msgs.msg
import time, math, sys, copy
from kbhit2 import TKBHit

StateToStr= {getattr(actionlib_msgs.msg.GoalStatus,key):key for key in ('PENDING', 'ACTIVE', 'RECALLED', 'REJECTED', 'PREEMPTED', 'ABORTED', 'SUCCEEDED', 'LOST')}
print 'StateToStr=',StateToStr

x_curr= None
q_curr= None
dq_curr= None

def JointStatesCallback(msg):
  global x_curr, q_curr, dq_curr
  x_curr= msg
  q_curr= x_curr.position
  dq_curr= x_curr.velocity
  #print 'x=%r'%(x_curr)

if __name__=='__main__':
  rospy.init_node('moto_test')

  joint_names= rospy.get_param('controller_joint_names')

  sub= rospy.Subscriber('/joint_states', sensor_msgs.msg.JointState, JointStatesCallback)

  client= actionlib.SimpleActionClient('/joint_trajectory_action', control_msgs.msg.FollowJointTrajectoryAction)

  # Wait some seconds for the head action server to start or exit
  if not client.wait_for_server(rospy.Duration(5.0)):
    rospy.logerr('Exiting - Joint Trajectory Action Server Not Found')
    rospy.signal_shutdown('Action Server not found')
    sys.exit(1)

  client.cancel_goal()

  def add_point(goal, time, positions, velocities):
    point= trajectory_msgs.msg.JointTrajectoryPoint()
    point.positions= copy.deepcopy(positions)
    point.velocities= copy.deepcopy(velocities)
    point.time_from_start= rospy.Duration(time)
    goal.trajectory.points.append(point)

  def make_traj(step, dt=1.0, with_current=True):
    goal= control_msgs.msg.FollowJointTrajectoryGoal()
    goal.trajectory.joint_names= joint_names
    if with_current:
      angles= copy.deepcopy(q_curr)
      add_point(goal, 0.0, angles, [0.0]*6)
    add_point(goal, dt*1.0, [step]*6, [0.0]*6)
    add_point(goal, dt*2.0, [0.0]*6, [0.0]*6)
    return goal

  with TKBHit() as kbhit:
    rate_adjustor= rospy.Rate(50)
    goal= make_traj(0.2)
    goal.trajectory.header.stamp= rospy.Time.now()
    client.send_goal(goal)
    print 'trajectory started. hit space to change the trajectory.'
    while not rospy.is_shutdown():
      c= kbhit.KBHit()
      if c==' ':
        print 'changing the trajectory...'
        goal= make_traj(-0.2, with_current=True)  #ERROR: Trajectory start position doesn't match current robot position (3011)
        #goal= make_traj(-0.2, with_current=False)  #ERROR: Validation failed: Trajectory doesn't start at current position
        goal.trajectory.header.stamp= rospy.Time.now()
        client.send_goal(goal)
        break
      if client.get_state() != actionlib_msgs.msg.GoalStatus.ACTIVE:
        print 'state: {} ({})'.format(StateToStr[client.get_state()], client.get_state())
      if client.get_state() not in (actionlib_msgs.msg.GoalStatus.PENDING, actionlib_msgs.msg.GoalStatus.ACTIVE):
        break
      rate_adjustor.sleep()
  print client.get_result()

  rospy.signal_shutdown('Done.')
