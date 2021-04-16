#!/usr/bin/python
#\file    draw_ellipse1.py
#\brief   Draw ellipse.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Apr.15, 2021

import sys
import random
from PyQt4 import QtCore,QtGui

def Print(*s):
  for ss in s:  print ss,
  print ''

class TEllipse(QtGui.QWidget):
  def __init__(self, parent=None):
    super(TEllipse, self).__init__(parent)

    self.changeColor()
    self.setBackgroundRole(QtGui.QPalette.Base)
    #self.setAutoFillBackground(True)

    size_policy= QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
    size_policy.setHeightForWidth(True)
    self.setSizePolicy(size_policy)

  def changeColor(self):
    self.color= QtGui.QColor(255*random.random(),255*random.random(),255*random.random())

  def minimumSizeHint(self):
    return QtCore.QSize(100, self.heightForWidth(100))

  def sizeHint(self):
    return QtCore.QSize(400, self.heightForWidth(400))

  def heightForWidth(self, width):
    return width*1.2

  #def resizeEvent(self, e):
    ##self.setMinimumWidth(self.height())
    #self.resize(QtCore.QSize(self.width(),self.heightForWidth(self.width())))

  def paintEvent(self, event):
    rect= QtCore.QRect(10, 10, self.width()-20, self.height()-20)
    #Always draw circle:
    #rad= min(self.width()-20, self.height()-20)
    #rect= QtCore.QRect((self.width()-rad)/2, (self.height()-rad)/2, rad, rad)

    linear_gradient= QtGui.QLinearGradient(0, 0, self.width(), self.height())
    linear_gradient.setColorAt(0.0, QtCore.Qt.white)
    #linear_gradient.setColorAt(0.2, QtCore.Qt.green)
    linear_gradient.setColorAt(0.2, self.color)
    linear_gradient.setColorAt(0.8, QtGui.QColor(0.6*self.color.red(),0.6*self.color.green(),0.6*self.color.blue()))
    linear_gradient.setColorAt(1.0, QtCore.Qt.black)

    painter= QtGui.QPainter(self)
    painter.setPen(QtCore.Qt.SolidLine)
    painter.setBrush(linear_gradient)
    painter.setRenderHint(QtGui.QPainter.Antialiasing)

    painter.save()
    painter.translate(0, 0)

    painter.drawEllipse(rect)

    painter.restore()

    painter.setPen(self.palette().dark().color())
    painter.setBrush(QtCore.Qt.NoBrush)
    painter.drawRect(QtCore.QRect(0, 0, self.width() - 1, self.height() - 1))


class TDrawEllipse(QtGui.QWidget):
  def __init__(self):
    QtGui.QWidget.__init__(self)
    self.InitUI()

  def InitUI(self):
    # Set window size.
    self.resize(320, 240)

    # Set window title
    self.setWindowTitle('DrawEllipse')


    mainlayout= QtGui.QGridLayout()

    self.ellipse= TEllipse()
    mainlayout.addWidget(self.ellipse,0,0)

    btn1= QtGui.QPushButton('Change color', self)
    btn1.clicked.connect(lambda:(self.ellipse.changeColor(),self.ellipse.update()))
    btn1.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
    self.btn1= btn1
    mainlayout.addWidget(self.btn1,0,1)

    btn2= QtGui.QPushButton('Exit', self)
    btn2.clicked.connect(lambda:self.close())
    btn2.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
    self.btn2= btn2
    mainlayout.addWidget(self.btn2,1,0,1,2)

    self.setLayout(mainlayout)

    # Show window
    self.show()



# Create an PyQT4 application object.
a = QtGui.QApplication(sys.argv)

# The QWidget widget is the base class of all user interface objects in PyQt4.
w = TDrawEllipse()

sys.exit(a.exec_())
