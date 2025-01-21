#!/usr/bin/python3
#\file    ros_ch_col.py
#\brief   Change colors of images from a ROS image topic.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Oct.14, 2024

import roslib; roslib.load_manifest('sensor_msgs')
import rospy
import sensor_msgs.msg
from cv_bridge import CvBridge, CvBridgeError
import cv2
import sys, os
import numpy as np

# Scale (min,max,step) space to (0,Max,1) space.
class TScaler(object):
  def __init__(self, min, max, step):
    self.min= min
    self.max= max
    self.step= step
    self.Max= int((max-min)/step)
  def To(self, value):
    return min(self.Max,max(0,int((value-self.min)/self.step)))
  def From(self, pos):
    return self.min+pos*self.step

#Get the image encoding of a ROS image topic.
#If convert_cv is true, the encoding is converted for OpenCV image conversion.
def GetImageEncoding(img_topic, convert_cv=False, time_out=5.0):
  try:
    msg= rospy.wait_for_message(img_topic, sensor_msgs.msg.Image, time_out)
    encoding= msg.encoding
    if not convert_cv:  return encoding
    if encoding=="rgb8":  return "bgr8"
    if encoding=="RGB8":  return "BGR8"
    #TODO: Add more conversion if necessary.
    return encoding;
  except (rospy.ROSException, rospy.ROSInterruptException):
    raise Exception('Failed to receive the image topic: {}'.format(img_topic))

def ImageCallback(msg, fmt, f_img_op):
  img= CvBridge().imgmsg_to_cv2(msg, fmt)
  if fmt=='16UC1':
    img_viz= cv2.cvtColor((img).astype('uint8'), cv2.COLOR_GRAY2BGR)
    #print np.min(img),np.max(img),'-->',np.min(img_viz),np.max(img_viz)
  else:  img_viz= img
  img_modified= f_img_op(img_viz)
  cv2.imshow('original', img_viz)
  cv2.imshow('image', img_modified)
  key= cv2.waitKey(1)&0xFF
  if key==ord('s'):
    for i in range(10000):
      filename= '/tmp/img{:05d}({}).png'.format(i,fmt)
      if not os.path.exists(filename):  break
    #cv2.imwrite(filename, img_viz)
    cv2.imwrite(filename, img_modified)
    print('Saved image into: {} ({})'.format(filename,fmt))
    #NOTE: Use cv2.imread(filename, cv2.IMREAD_ANYDEPTH) to read depth image (16UC1).
  elif key==ord('q'):
    rospy.signal_shutdown('quit.')
    cv2.destroyAllWindows()

class TChangeImgColor(object):
  def __init__(self):
    win_name= 'image'
    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, self.OnMouse)
    self.add_color= TScaler(-2.0, 2.0, 0.002)
    self.mult_color= TScaler(-5.0, 5.0, 0.1)
    self.add_b,self.add_g,self.add_r= 0.04, 0.52, 0.72
    self.mult_b,self.mult_g,self.mult_r= 1.0,1.0,1.0
    cv2.createTrackbar('AddB', win_name, self.add_color.To(self.add_b), self.add_color.Max,
                       lambda pos:setattr(self,'add_b',self.add_color.From(pos)))
    cv2.createTrackbar('AddG', win_name, self.add_color.To(self.add_g), self.add_color.Max,
                       lambda pos:setattr(self,'add_g',self.add_color.From(pos)))
    cv2.createTrackbar('AddR', win_name, self.add_color.To(self.add_r), self.add_color.Max,
                       lambda pos:setattr(self,'add_r',self.add_color.From(pos)))
    cv2.createTrackbar('MultB', win_name, self.mult_color.To(self.mult_b), self.mult_color.Max,
                       lambda pos:setattr(self,'mult_b',self.mult_color.From(pos)))
    cv2.createTrackbar('MultG', win_name, self.mult_color.To(self.mult_g), self.mult_color.Max,
                       lambda pos:setattr(self,'mult_g',self.mult_color.From(pos)))
    cv2.createTrackbar('MultR', win_name, self.mult_color.To(self.mult_r), self.mult_color.Max,
                       lambda pos:setattr(self,'mult_r',self.mult_color.From(pos)))

  def OnMouse(self, event, x, y, flags, param):
    if event==cv2.EVENT_LBUTTONUP:
      print('Add=',self.add_b,self.add_g,self.add_r)
      print('Mult=',self.mult_b,self.mult_g,self.mult_r)

  def ChangeImgColor(self, img):
    # Normalize the image to the range [0, 1] for easier manipulation
    img_float= img.astype(np.float32) / 255.0

    ## Define the target color in normalized BGR
    #trg_col= np.array([self.add_b,self.add_g,self.add_r], dtype='float32')

    ## Calculate the brightness of each pixel (using the mean of BGR channels)
    #brightness= np.mean(img_float, axis=2, keepdims=True)  # Shape (H, W, 1)

    ## Define how much dark colors should shift to the target color based on brightness
    ## The formula will blend the image's colors with the target color, based on brightness
    ## The lower the brightness, the more the color shifts toward the target color
    #converted_img= img_float * brightness + trg_col * (1 - brightness)

    converted_img= img_float * np.array([self.mult_b,self.mult_g,self.mult_r], dtype='float32') + np.array([self.add_b,self.add_g,self.add_r], dtype='float32')

    # Convert back to the 0-255 range and uint8 type for saving/display
    output_img= (np.clip(converted_img,0.0,1.0) * 255).astype(np.uint8)
    return output_img

if __name__=='__main__':
  topic= sys.argv[1] if len(sys.argv)>1 else '/camera/color/image_raw'
  fmt= sys.argv[2] if len(sys.argv)>2 else None
  rospy.init_node('ros_ch_col')
  if fmt is None:
    #fmt= '16UC1' if 'depth' in topic else 'bgr8'
    fmt= GetImageEncoding(topic, convert_cv=True)

  ch_img_col= TChangeImgColor()
  f_img_op= ch_img_col.ChangeImgColor

  image_sub= rospy.Subscriber(topic, sensor_msgs.msg.Image, lambda msg:ImageCallback(msg,fmt,f_img_op))
  rospy.spin()
