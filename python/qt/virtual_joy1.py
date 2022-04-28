#!/usr/bin/python
#\file    virtual_joy1.py
#\brief   Virtual joystick example.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Apr.26, 2022

#Ref. https://stackoverflow.com/questions/55876713/how-to-create-a-joystick-controller-widget-in-pyqt/55899694#55899694

import sys
from PyQt4 import QtCore,QtGui

class TVirtualJoyStick(QtGui.QWidget):
  def __init__(self, *args, **kwargs):
    super(TVirtualJoyStick, self).__init__(*args, **kwargs)
    self.setMinimumSize(100, 100)

    self.ResetStickPos()
    self.stick_grabbed= False
    self.stick_rad= 30

  def ResetStickPos(self):
    #Normalized stick position:
    self.stick_pos= QtCore.QLineF(QtCore.QPointF(0,0),QtCore.QPointF(0,0))

  def Center(self):
    return QtCore.QPointF(self.width()/2, self.height()/2)

  def paintEvent(self, event):
    painter= QtGui.QPainter(self)
    bounds= QtCore.QRectF(0, 0, self.width()-1, self.height()-1)
    painter.drawEllipse(bounds)
    stick_displacement= self.Center()+QtCore.QPointF(self.stick_pos.p2().x()*self.width()/2.0,-self.stick_pos.p2().y()*self.height()/2.0)
    bounds= QtCore.QRectF(-self.stick_rad, -self.stick_rad, self.stick_rad*2, self.stick_rad*2).translated(stick_displacement)
    painter.setBrush(QtCore.Qt.black)
    painter.drawEllipse(bounds)

  def mousePressEvent(self, event):
    dp= QtCore.QLineF(self.Center(), event.pos())
    self.stick_grabbed= dp.length()<self.stick_rad
    if self.stick_grabbed:  self.pos_grabbed= event.pos()
    return super(TVirtualJoyStick, self).mousePressEvent(event)

  def mouseReleaseEvent(self, event):
    self.stick_grabbed= False
    self.ResetStickPos()
    self.update()

  def mouseMoveEvent(self, event):
    if self.stick_grabbed:
      dp= QtCore.QPointF(event.pos()-self.pos_grabbed)
      self.stick_pos= QtCore.QLineF(QtCore.QPointF(0,0),
                                    QtCore.QPointF(
                                      dp.x()/self.width()*2.0,
                                      -dp.y()/self.height()*2.0))
      if self.stick_pos.length()>1.0:  self.stick_pos.setLength(1.0)
      #print 'debug', self.stick_pos, self.stick_pos.p2(), self.stick_pos.length(), self.stick_pos.angle()
      self.update()
    print self.stick_pos.p2(), self.stick_pos.length(), self.stick_pos.angle()


def Print(*s):
  for ss in s:  print ss,
  print ''

class TVirtualJoyStickTest(QtGui.QWidget):
  def __init__(self):
    QtGui.QWidget.__init__(self)
    self.InitUI()

  def InitUI(self):
    # Set window size.
    self.resize(400, 240)

    # Set window title
    self.setWindowTitle('VirtualJoyStickTest')

    #mainlayout= QtGui.QVBoxLayout()
    #self.setLayout(mainlayout)

    joystick1= TVirtualJoyStick(self)
    joystick1.move(40, 70)
    self.joystick1= joystick1
    #mainlayout.addWidget(joystick1)

    # Add a button
    btn1= QtGui.QPushButton('_________Exit?_________', self)
    #btn1.setFlat(True)
    btn1.setToolTip('Click to exit')
    #btn1.clicked.connect(lambda:self.close() if self.slider1.value()<1500 else Print('Hint: Set value less than 1500 to exit'))
    btn1.clicked.connect(lambda:self.close())
    btn1.move(170, 120)
    self.btn1= btn1
    #mainlayout.addWidget(btn1)

    # Show window
    self.show()

    print 'debug',joystick1.size(), joystick1.size().width(), joystick1.size().height()

# Create an PyQT4 application object.
a = QtGui.QApplication(sys.argv)

# The QWidget widget is the base class of all user interface objects in PyQt4.
w = TVirtualJoyStickTest()

sys.exit(a.exec_())

