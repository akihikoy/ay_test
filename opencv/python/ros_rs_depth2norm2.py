#!/usr/bin/python
#\file    ros_rs_depth2norm2.py
#\brief   Convert a depth image to a normal image with considering 3D geometry.
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
from depth2normal2 import DepthToNormal

def ImageCallback(msg, fmt):
  img_depth= CvBridge().imgmsg_to_cv2(msg, fmt)
  proj_mat= np.array([[612.449462890625, 0.0, 317.5238952636719, 0.0], [0.0, 611.5702514648438, 237.89498901367188, 0.0], [0.0, 0.0, 1.0, 0.0]])

  norm_alpha,norm_beta= DepthToNormal(img_depth, proj_mat, resize_ratio=0.25)
  beta_img= ((1.0-norm_beta/(0.5*np.pi))*255.).astype('uint8')
  hsvimg= np.dstack(((norm_alpha/np.pi*127.+128.).astype('uint8'),
                     (np.ones_like(norm_alpha)*255).astype('uint8'),
                     beta_img,
                     ))

  cv2.imshow('depth',cv2.cvtColor(img_depth.astype('uint8'), cv2.COLOR_GRAY2BGR))
  cv2.imshow('normal',cv2.cvtColor(hsvimg, cv2.COLOR_HSV2BGR))
  #beta_img[beta_img<220]= 0
  cv2.imshow('normal(beta)',beta_img)

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
