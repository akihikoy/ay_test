#!/usr/bin/python
#\file    label2.py
#\brief   Expanding QLabel.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Apr.14, 2021

import sys
from PyQt4 import QtCore,QtGui

def Print(*s):
  for ss in s:  print ss,
  print ''

class TLabel(QtGui.QWidget):
  def __init__(self):
    QtGui.QWidget.__init__(self)
    self.InitUI()

  def InitUI(self):
    # Set window size.
    self.resize(620, 520)

    # Set window title
    self.setWindowTitle("Label")

    mainlayout= QtGui.QVBoxLayout()
    #mainlayout= QtGui.QGridLayout()
    self.setLayout(mainlayout)

    label1= QtGui.QLabel('Label label\nLabel\nhoge hoge hoge hoge hoge\nhoge hoge\naaa', self)
    label1.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
    label1.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
    #label1.move(10, 60)
    label1.font_size= (8,28)
    label1.setFont(QtGui.QFont('', label1.font_size[0]))
    label1.resizeEvent= lambda event,obj=label1: self.ResizeText(obj,event)
    mainlayout.addWidget(label1)#,0,1)

    vspacer1= QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
    mainlayout.addSpacerItem(vspacer1)#,0,1)

    btn1= QtGui.QPushButton('Exit', self)
    #btn1.setFlat(True)
    btn1.clicked.connect(lambda:self.close())
    btn1.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
    btn1.resize(btn1.sizeHint())
    #btn1.move(100, 150)
    btn1.font_size= (10,30)
    btn1.setFont(QtGui.QFont('', btn1.font_size[0]))
    btn1.resizeEvent= lambda event,obj=btn1: self.ResizeText(obj,event)
    self.btn1= btn1
    mainlayout.addWidget(btn1)#,0,1)

    # Show window
    self.show()

  def ResizeText(self, obj, event):
    font_size= min(obj.font_size[1],max(obj.font_size[0],int(self.rect().height()/100.*obj.font_size[0])))
    f= QtGui.QFont('', font_size)
    if isinstance(obj,QtGui.QRadioButton):
      obj.setStyleSheet('QRadioButton::indicator {{width:{0}px;height:{0}px;}};'.format(1.3*font_size))
    obj.setFont(f)

# Create an PyQT4 application object.
a = QtGui.QApplication(sys.argv)

# The QWidget widget is the base class of all user interface objects in PyQt4.
w = TLabel()

sys.exit(a.exec_())

