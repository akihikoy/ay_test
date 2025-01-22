#!/usr/bin/python3
#\file    ros_cv.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Mar.03, 2017
import roslib; roslib.load_manifest('sensor_msgs')
import sys
import rospy
import cv2
import sensor_msgs.msg
from cv_bridge import CvBridge, CvBridgeError

class image_converter:
  def __init__(self):
    self.bridge= CvBridge()
    self.image_sub= rospy.Subscriber("/camera/color/image_raw",sensor_msgs.msg.Image,self.callback)

  def callback(self,data):
    try:
      cv_image= self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)
      return

    (rows,cols,channels)= cv_image.shape
    if cols>60 and rows>60:
      cv2.circle(cv_image, (50,50), 10, 255)

    cv2.imshow("camera", cv_image)
    c= cv2.waitKey(3)
    if c&127 in (ord('q'),27):  #'q' or ESC
      self.image_sub.unregister()
      rospy.signal_shutdown('quit.')
      cv2.destroyAllWindows()

if __name__ == '__main__':
  ic= image_converter()
  rospy.init_node('ros_cv', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
