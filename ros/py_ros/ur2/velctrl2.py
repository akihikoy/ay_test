#!/usr/bin/python
#\file    velctrl2.py
#\brief   Velocity control sample.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Oct.12, 2020

'''
Src: https://github.com/UniversalRobots/Universal_Robots_ROS_Driver/issues/231#issuecomment-688415091

NOTE:
Executing the following commands is necessary:
$ rosservice call /controller_manager/switch_controller "stop_controllers: ['scaled_pos_joint_traj_controller']"
$ rosservice call /controller_manager/switch_controller "start_controllers: ['joint_group_vel_controller']"

Exiting from the velocity control mode, run:
$ rosservice call /controller_manager/switch_controller "stop_controllers: ['joint_group_vel_controller']"
$ rosservice call /controller_manager/switch_controller "start_controllers: ['scaled_pos_joint_traj_controller']"
'''

import signal
import sys

import roslib
import rospy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState


class RobotMover(object):
    """Small node to move the robot"""

    def __init__(self, speed):
        super(RobotMover, self).__init__()

        rospy.init_node('velocity_node')
        self.pub = rospy.Publisher('/joint_group_vel_controller/command',
                                   Float64MultiArray, queue_size=10)
        self.msg = Float64MultiArray()
        print "Speed is {}".format(speed)
        self.msg.data = [0, 0, 0, 0, 0, speed]
        self.msg.layout.data_offset = 1

    def spin_once(self):
        self.pub.publish(self.msg)

    def signal_handler(self, sig, frame):
        print 'You pressed Ctrl+C!'
        self.msg.data = [0, 0, 0, 0, 0, 0.0]
        self.pub.publish(self.msg)
        sys.exit(0)


if __name__ == "__main__":
    try:
        speed = float(sys.argv[1])
    except IndexError as err:
        print "No speed given."
        sys.exit(1)
    except ValueError as err:
        print "No valid speed given"
        sys.exit(1)

    robot_mover = RobotMover(speed)
    rate = rospy.Rate(10)

    signal.signal(signal.SIGINT, robot_mover.signal_handler)

    while not rospy.is_shutdown():
        robot_mover.spin_once()
        rate.sleep()

