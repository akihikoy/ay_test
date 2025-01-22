#!/usr/bin/python3
#\file    cv_qt1.py
#\brief   Displaying OpenCV images in Qt framework.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Sep.26, 2023

import sys
#from PyQt4 import QtCore,QtGui
from _import_qt import *
import cv2

class TImgDialog(QtGui.QDialog):
  def __init__(self, parent=None, cam_dev=0):
    super(TImgDialog, self).__init__(parent)
    self.cap= cv2.VideoCapture(cam_dev)
    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    print(f'Camera({cam_dev}) opened')
    print('Press q to quit.')
    #self.resize(321, 241)
    self.show()

  def captureImage(self):
    ret,frame= self.cap.read()
    if frame is None:
      self.cv_img= None
      print('WARNING: Failed to capture')
      return
    self.cv_img= frame
    height, width, byte_value = self.cv_img.shape
    byte_value= byte_value * width
    cv2.cvtColor(self.cv_img, cv2.COLOR_BGR2RGB, self.cv_img)
    self.q_img = QtGui.QImage(self.cv_img, width, height, byte_value, QtGui.QImage.Format_RGB888)

  def paintEvent(self, event):
    self.captureImage()
    if self.cv_img is None:
      return
    self.resize(self.cv_img.shape[1], self.cv_img.shape[0])
    painter= QtGui.QPainter()
    painter.begin(self)
    painter.drawImage(0, 0, self.q_img)
    painter.end()

  def keyPressEvent(self, event):
    super(TImgDialog, self).keyPressEvent(event)
    key= event.text()
    if key==' ':
      self.update()
    elif key=='q':
      self.close()

# Create an PyQT4 application object.
a = QtGui.QApplication(sys.argv)

# The QWidget widget is the base class of all user interface objects in PyQt4.
w = TImgDialog(cam_dev=int(sys.argv[1]) if len(sys.argv)>1 else 0)

sys.exit(a.exec_())
