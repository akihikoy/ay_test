#!/usr/bin/python3
#\file    slider1.py
#\brief   Test QSlider
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
    self.resize(320, 240)

    # Set window title
    self.setWindowTitle("Slider")

    # Add a button
    btn1= QtGui.QPushButton('_________Exit?_________', self)
    #btn1.setFlat(True)
    btn1.setToolTip('Click to make something happen')
    btn1.clicked.connect(lambda:self.close() if self.slider1.toValue()<1500 else Print('Hint: Set value less than 1500 to exit'))
    btn1.resize(btn1.sizeHint())
    btn1.move(100, 150)
    self.btn1= btn1

    label1= QtGui.QLabel('0',self)
    label1.move(120, 60)
    self.label1= label1

    #slider1= QtGui.QSlider(QtCore.Qt.Vertical, self)
    slider1= QtGui.QSlider(QtCore.Qt.Horizontal, self)
    slider1.setTickPosition(QtGui.QSlider.TicksBothSides)
    slider1.setRange(0, 8)
    slider1.setTickInterval(1)
    slider1.setSingleStep(1)
    slider1.setValue(6)
    slider1.move(10, 60)
    slider1.resize(100, 20)
    slider1.toValue= lambda: 1000 + 100*self.slider1.value()
    slider1.valueChanged.connect(lambda:self.label1.setText(str(self.slider1.toValue())))
    self.slider1= slider1

    self.label1.setText(str(self.slider1.toValue()))

    # Show window
    self.show()

# Create an PyQT4 application object.
a = QtGui.QApplication(sys.argv)

# The QWidget widget is the base class of all user interface objects in PyQt4.
w = TSlider()

sys.exit(a.exec_())

