#!/usr/bin/python
#\file    velctrl1.py
#\brief   Velocity control ver.1 (waiving).
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Nov.26, 2019

import roslib
import rospy
import kortex_driver.msg
import kortex_driver.srv
import math
from kbhit2 import TKBHit

if __name__=='__main__':
  rospy.init_node('gen3_test')
  rospy.wait_for_service('/gen3a/base/send_joint_speeds_command')
  srv_joint_speeds= rospy.ServiceProxy('/gen3a/base/send_joint_speeds_command', kortex_driver.srv.SendJointSpeedsCommand)

  joint_names= ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'joint_7']


  speed_req= kortex_driver.srv.SendJointSpeedsCommandRequest()
  #NOTE: JointSpeed/value is in DEGREES per second.
  #cf. https://github.com/Kinovarobotics/kortex/blob/master/api_cpp/doc/markdown/messages/Base/JointSpeed.md
  rad2deg= lambda q:q/math.pi*180.0
  for jidx,jname in enumerate(joint_names):
    joint_speed= kortex_driver.msg.JointSpeed()
    joint_speed.joint_identifier= jidx
    joint_speed.value= 0.0
    speed_req.input.joint_speeds.append(joint_speed)

  t0= rospy.Time.now()
  rate= rospy.Rate(200)

  kbhit= TKBHit()
  try:
    while not rospy.is_shutdown():
      if kbhit.IsActive():
        key= kbhit.KBHit()
        if key is not None:  #Press any key to stop.
          break;
      else:
        break

      t= (rospy.Time.now()-t0).to_sec()
      for joint_speed in speed_req.input.joint_speeds:
        #NOTE: JointSpeed/value is in DEGREES per second.
        joint_speed.value= rad2deg(0.08*math.sin(math.pi*t))
      srv_joint_speeds.call(speed_req)
      rate.sleep()

  except KeyboardInterrupt:
    print 'Interrupted'

  finally:
    kbhit.Deactivate()
    #To make sure the robot stops:
    speed_req= kortex_driver.srv.SendJointSpeedsCommandRequest()
    for jidx,jname in enumerate(joint_names):
      joint_speed= kortex_driver.msg.JointSpeed()
      joint_speed.joint_identifier= jidx
      joint_speed.value= 0.0
      speed_req.input.joint_speeds.append(joint_speed)
    srv_joint_speeds.call(speed_req)
