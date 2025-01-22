#!/usr/bin/python3
#\file    ros_py3.py
#\brief   ROS with Python3, testing with image subscribing.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.27, 2021
'''
$ roscore
$ ./ros_pub_img.py
$ ./ros_py3.py
'''

import roslib; roslib.load_manifest('sensor_msgs')
import rospy
import sensor_msgs.msg
#NOTE: cv_bridge should be built for Pytho3 to use (not available in binary).
#from cv_bridge import CvBridge, CvBridgeError
#NOTE: OpenCV should be installed for Python3 (not available in apt???).
#import cv2
from PIL import Image as PILImage

def ImageCallback(msg):
  print(('received:',type(msg)))
  #img= CvBridge().imgmsg_to_cv2(msg, "bgr8")
  #img= cv2.flip(img, 1)
  #cv2.imshow('image', img)
  #if cv2.waitKey(1)&0xFF==ord('q'):
    #rospy.signal_shutdown('quit.')
    #cv2.destroyAllWindows()
  #NOTE: Since cv_bridge is not available, we use PIL instead.
  encoding_ros_to_pil= {'mono8':'L', 'rgb8':'RGB', 'bgr8':'BGR', 'rgba8':'RGBA', 'yuv422':'YCbCr'}
  pil= PILImage.frombytes('RGB', (msg.width, msg.height),
                msg.data, 'raw', encoding_ros_to_pil[msg.encoding], 0, 1)
  filename= '/tmp/frame.png'
  pil.save(filename)
  print(('  image saved as:', filename))

if __name__=='__main__':
  rospy.init_node('ros_sub_img')
  image_sub= rospy.Subscriber("/camera/color/image_raw", sensor_msgs.msg.Image, ImageCallback)
  rospy.spin()
