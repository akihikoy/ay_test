#!/usr/bin/python
#\file    draw_primitive1.py
#\brief   Test of drawing a primitive widget.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Apr.19, 2021

import sys
import random
from PyQt4 import QtCore,QtGui

def Print(*s):
  for ss in s:  print ss,
  print ''

class TRenderPrimitive(QtGui.QWidget):
  def __init__(self, shape, margin, color, parent=None):
    super(TRenderPrimitive, self).__init__(parent)

    self.shape= shape
    self.margin= margin  #(horizontal_margin(ratio),vertical_margin(ratio))
    self.color= color
    self.min_size= 100
    self.max_size= 400
    self.draw_bevel= True
    self.setBackgroundRole(QtGui.QPalette.Base)
    #self.setAutoFillBackground(True)

    size_policy= QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
    size_policy.setHeightForWidth(True)
    self.setSizePolicy(size_policy)

  def setRandomColor(self):
    self.color= [255*random.random(),255*random.random(),255*random.random()]

  def setShape(self, shape):
    self.shape= shape

  def setMargin(self, margin):
    self.margin= margin

  def minimumSizeHint(self):
    return QtCore.QSize(self.min_size, self.heightForWidth(self.min_size))

  def sizeHint(self):
    return QtCore.QSize(self.max_size, self.heightForWidth(self.max_size))

  def heightForWidth(self, width):
    return width*1.2

  def paintEvent(self, event):
    col1= QtGui.QColor(*self.color)
    col2= QtGui.QColor(0.6*self.color[0], 0.6*self.color[1], 0.6*self.color[2])
    linear_gradient= QtGui.QLinearGradient(0, 0, self.width(), self.height())
    linear_gradient.setColorAt(0.0, QtCore.Qt.white)
    linear_gradient.setColorAt(0.2, col1)
    linear_gradient.setColorAt(0.8, col2)
    linear_gradient.setColorAt(1.0, QtCore.Qt.black)

    painter= QtGui.QPainter(self)
    painter.setPen(QtCore.Qt.SolidLine)
    painter.setBrush(linear_gradient)
    painter.setRenderHint(QtGui.QPainter.Antialiasing)

    painter.save()
    painter.translate(0, 0)

    if self.shape in ('ellipse','rect'):
      rect= QtCore.QRect(self.width()*self.margin[0], self.height()*self.margin[1], self.width()*(1.0-2.0*self.margin[0]), self.height()*(1.0-2.0*self.margin[1]))
    elif self.shape in ('circle','square'):
      l= min(self.width()*(1.0-2.0*self.margin[0]), self.height()*(1.0-2.0*self.margin[1]))
      rect= QtCore.QRect((self.width()-l)/2, (self.height()-l)/2, l, l)

    if self.shape in ('ellipse','circle'):
      painter.drawEllipse(rect)
    elif self.shape in ('rect','square'):
      painter.drawRect(rect)

    painter.restore()

    if self.draw_bevel:
      painter.setPen(self.palette().dark().color())
      painter.setBrush(QtCore.Qt.NoBrush)
      painter.drawRect(QtCore.QRect(0, 0, self.width() - 1, self.height() - 1))


class TDrawPrimitive(QtGui.QWidget):
  def __init__(self):
    QtGui.QWidget.__init__(self)
    self.InitUI()

  def InitUI(self):
    # Set window size.
    self.resize(320, 240)

    # Set window title
    self.setWindowTitle('DrawEllipse')

    mainlayout= QtGui.QGridLayout()

    cmbbx1= QtGui.QComboBox(self)
    cmbbx1.addItem('rect')
    cmbbx1.addItem('ellipse')
    cmbbx1.addItem('square')
    cmbbx1.addItem('circle')
    cmbbx1.setCurrentIndex(0)
    cmbbx1.activated[str].connect(lambda:(self.primitive.setShape(self.cmbbx1.currentText()), self.primitive.update()))
    self.cmbbx1= cmbbx1
    mainlayout.addWidget(self.cmbbx1,0,0,1,2)

    self.primitive= TRenderPrimitive('rect',(0.05,0.05),(0,0,0),self)
    self.primitive.setRandomColor()
    mainlayout.addWidget(self.primitive,1,0)

    btn1= QtGui.QPushButton('Change color', self)
    btn1.clicked.connect(lambda:(self.primitive.setRandomColor(),self.primitive.update()))
    btn1.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
    self.btn1= btn1
    mainlayout.addWidget(self.btn1,1,1)

    btn2= QtGui.QPushButton('Exit', self)
    btn2.clicked.connect(lambda:self.close())
    btn2.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
    self.btn2= btn2
    mainlayout.addWidget(self.btn2,2,0,1,2)

    self.setLayout(mainlayout)

    # Show window
    self.show()


if __name__=='__main__':
  # Create an PyQT4 application object.
  a = QtGui.QApplication(sys.argv)

  # The QWidget widget is the base class of all user interface objects in PyQt4.
  w = TDrawPrimitive()

  sys.exit(a.exec_())
