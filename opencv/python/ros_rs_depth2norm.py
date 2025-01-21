#!/usr/bin/python3
#\file    ros_rs_depth2norm.py
#\brief   Convert a depth image to a normal image.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Oct.31, 2022
import roslib; roslib.load_manifest('sensor_msgs')
import rospy
import sensor_msgs.msg
from cv_bridge import CvBridge, CvBridgeError
import cv2
import sys, os
import numpy as np
from ros_img_save import GetImageEncoding
from depth2normal import DepthToNormalImg

def ImageCallback(msg, fmt):
  img_depth= CvBridge().imgmsg_to_cv2(msg, fmt)

  img_norm,img_amp= DepthToNormalImg(img_depth, with_amp=True)

  cv2.imshow('depth',cv2.cvtColor(img_depth.astype('uint8'), cv2.COLOR_GRAY2BGR))
  cv2.imshow('normal(abs)',np.abs(img_norm))
  cv2.imshow('normal(amp)',img_amp.astype('uint8'))
  img_amp= cv2.GaussianBlur(img_amp,(5,5),0)
  cv2.imshow('normal(amp-blur)',img_amp.astype('uint8'))

  #if fmt=='16UC1':
    #img_viz= cv2.cvtColor((img).astype('uint8'), cv2.COLOR_GRAY2BGR)
    ##print np.min(img),np.max(img),'-->',np.min(img_viz),np.max(img_viz)
  #else:  img_viz= img
  #cv2.imshow('image', img_viz)

  key= cv2.waitKey(1)&0xFF
  if key==ord('q'):
    rospy.signal_shutdown('quit.')
    cv2.destroyAllWindows()

if __name__=='__main__':
  topic= sys.argv[1] if len(sys.argv)>1 else '/camera/aligned_depth_to_color/image_raw'
  fmt= sys.argv[2] if len(sys.argv)>2 else None
  rospy.init_node('ros_img_save')
  if fmt is None:
    fmt= GetImageEncoding(topic, convert_cv=True)
  image_sub= rospy.Subscriber(topic, sensor_msgs.msg.Image, lambda msg:ImageCallback(msg,fmt))
  rospy.spin()
