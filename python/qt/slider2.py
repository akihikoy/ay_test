#!/usr/bin/python3
#\file    slider2.py
#\brief   Slider with labels.
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
    slidergroup.addWidget(label1, 0, 5, 1, 1)

    self.label1.setText(str(self.slider1.toValue()))

    labelt1= QtGui.QLabel('1000',self)
    slidergroup.addWidget(labelt1, 1, 0, 1, 1, QtCore.Qt.AlignLeft)
    labelt2= QtGui.QLabel('1200',self)
    slidergroup.addWidget(labelt2, 1, 1, 1, 1, QtCore.Qt.AlignLeft)
    labelt3= QtGui.QLabel('1400',self)
    slidergroup.addWidget(labelt3, 1, 2, 1, 1, QtCore.Qt.AlignCenter)
    labelt4= QtGui.QLabel('1600',self)
    slidergroup.addWidget(labelt4, 1, 3, 1, 1, QtCore.Qt.AlignRight)
    labelt5= QtGui.QLabel('1800',self)
    slidergroup.addWidget(labelt5, 1, 4, 1, 1, QtCore.Qt.AlignRight)

    vspacer1= QtGui.QSpacerItem(10, 10, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
    slidergroup.addItem(vspacer1, 2, 0, 1, 6)


    # Add a button
    btn1= QtGui.QPushButton('_________Exit?_________', self)
    #btn1.setFlat(True)
    btn1.setToolTip('Click to make something happen')
    btn1.clicked.connect(lambda:self.close() if self.slider1.toValue()<1500 else Print('Hint: Set value less than 1500 to exit'))
    btn1.resize(btn1.sizeHint())
    #btn1.move(100, 150)
    self.btn1= btn1
    mainlayout.addWidget(btn1)

    # Show window
    self.show()

# Create an PyQT4 application object.
a = QtGui.QApplication(sys.argv)

# The QWidget widget is the base class of all user interface objects in PyQt4.
w = TSlider()

sys.exit(a.exec_())

