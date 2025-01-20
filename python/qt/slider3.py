#!/usr/bin/python3
#\file    slider3.py
#\brief   Expanding slider.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Apr.14, 2021

import sys
from PyQt4 import QtCore,QtGui

def Print(*s):
  for ss in s:  print(ss, end=' ')
  print('')

class TSlider(QtGui.QWidget):
  def __init__(self):
    QtGui.QWidget.__init__(self)
    self.InitUI()

  def InitUI(self):
    # Set window size.
    self.resize(320, 120)

    # Set window title
    self.setWindowTitle("Slider")

    mainlayout= QtGui.QVBoxLayout()
    self.setLayout(mainlayout)

    slidergroup= QtGui.QGridLayout()
    #slidergroup.move(10, 60)
    #slidergroup.resize(10, 60)
    #self.setLayout(slidergroup)
    mainlayout.addLayout(slidergroup)

    #slider1= QtGui.QSlider(QtCore.Qt.Vertical, self)
    slider1= QtGui.QSlider(QtCore.Qt.Horizontal, self)
    slider1.setTickPosition(QtGui.QSlider.TicksBothSides)
    slider1.setRange(0, 8)
    slider1.setTickInterval(1)
    slider1.setSingleStep(1)
    slider1.setValue(6)
    #slider1.move(10, 60)
    slider1.resize(100, 20)
    slider1.toValue= lambda: 1000 + 100*self.slider1.value()
    slider1.valueChanged.connect(lambda:self.label1.setText(str(self.slider1.toValue())))
    self.slider1= slider1
    slidergroup.addWidget(slider1, 0, 0, 1, 5)

    label1= QtGui.QLabel('0',self)
    self.label1= label1
    label1.font_size= (10,30)
    label1.setFont(QtGui.QFont('', label1.font_size[0]))
    #label1.resizeEvent= lambda event,obj=label1: self.ResizeText(obj,event)
    #NOTE: resizeEvent is defined below.
    slidergroup.addWidget(label1, 0, 5, 1, 1)

    self.label1.setText(str(self.slider1.toValue()))

    #NOTE: The resizeEvent of following labels do not happen when the window is resized.
    #  Thus, these events are included in that of label1.
    labelt1= QtGui.QLabel('1000',self)
    labelt1.font_size= (6,20)
    labelt1.setFont(QtGui.QFont('', labelt1.font_size[0]))
    #labelt1.resizeEvent= lambda event,obj=labelt1: self.ResizeText(obj,event)
    slidergroup.addWidget(labelt1, 1, 0, 1, 1, QtCore.Qt.AlignLeft)
    labelt2= QtGui.QLabel('1200',self)
    labelt2.font_size= (6,20)
    labelt2.setFont(QtGui.QFont('', labelt2.font_size[0]))
    #labelt2.resizeEvent= lambda event,obj=labelt2: self.ResizeText(obj,event)
    slidergroup.addWidget(labelt2, 1, 1, 1, 1, QtCore.Qt.AlignLeft)
    labelt3= QtGui.QLabel('1400',self)
    labelt3.font_size= (6,20)
    labelt3.setFont(QtGui.QFont('', labelt3.font_size[0]))
    #labelt3.resizeEvent= lambda event,obj=labelt3: self.ResizeText(obj,event)
    slidergroup.addWidget(labelt3, 1, 2, 1, 1, QtCore.Qt.AlignCenter)
    labelt4= QtGui.QLabel('1600',self)
    labelt4.font_size= (6,20)
    labelt4.setFont(QtGui.QFont('', labelt4.font_size[0]))
    #labelt4.resizeEvent= lambda event,obj=labelt4: self.ResizeText(obj,event)
    slidergroup.addWidget(labelt4, 1, 3, 1, 1, QtCore.Qt.AlignRight)
    labelt5= QtGui.QLabel('1800',self)
    labelt5.font_size= (6,20)
    labelt5.setFont(QtGui.QFont('', labelt5.font_size[0]))
    #labelt5.resizeEvent= lambda event,obj=labelt5: self.ResizeText(obj,event)
    slidergroup.addWidget(labelt5, 1, 4, 1, 1, QtCore.Qt.AlignRight)

    #NOTE: This event does not happen when only the height is changed.
    #  So, these are included in the reszie event of btn1.
    #label1.resizeEvent= lambda event,objs=(label1,labelt1,labelt2,labelt3,labelt4,labelt5): ([self.ResizeText(obj,event) for obj in objs]+[None])[-1]

    vspacer1= QtGui.QSpacerItem(10, 10, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
    slidergroup.addItem(vspacer1, 2, 0, 1, 6)


    # Add a button
    btn1= QtGui.QPushButton('_________Exit?_________', self)
    #btn1.setFlat(True)
    btn1.setToolTip('Click to make something happen')
    btn1.clicked.connect(lambda:self.close() if self.slider1.toValue()<1500 else Print('Hint: Set value less than 1500 to exit'))
    btn1.resize(btn1.sizeHint())
    #btn1.move(100, 150)
    btn1.font_size= (10,30)
    btn1.setFont(QtGui.QFont('', btn1.font_size[0]))
    #btn1.resizeEvent= lambda event,obj=btn1: self.ResizeText(obj,event)
    self.btn1= btn1
    mainlayout.addWidget(btn1)

    btn1.resizeEvent= lambda event,objs=(btn1,label1,labelt1,labelt2,labelt3,labelt4,labelt5): ([self.ResizeText(obj,event) for obj in objs]+[None])[-1]

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
w = TSlider()

sys.exit(a.exec_())

