#!/usr/bin/python3
#\file    virtual_joy2.py
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
    #self.setMinimumSize(100, 100)
    self.setBackgroundRole(QtGui.QPalette.Base)

    self.kind= 'circle'  #circle, ellipse
    self.bg_color= [200]*3
    self.resetStickPos()
    self.stick_grabbed= False
    self.stick_rad= 0.3  #Stick radius per width of movable range.
    self.stick_color= [128, 128, 255]

  def resetStickPos(self):
    #Normalized stick position:
    self.stick_pos= QtCore.QLineF(QtCore.QPointF(0,0),QtCore.QPointF(0,0))

  def center(self):
    return QtCore.QPointF(self.width()/2, self.height()/2)

  #def minimumSizeHint(self):
    #return QtCore.QSize(100, self.heightForWidth(100))

  #def sizeHint(self):
    #return QtCore.QSize(400, self.heightForWidth(400))

  #def heightForWidth(self, width):
    #return width*1.2

  def position(self):
    return self.stick_pos.p2(), self.stick_pos.length(), self.stick_pos.angle()

  def getGradient(self, color, bounds, reverse=False):
    col1= QtGui.QColor(*color)
    col2= QtGui.QColor(0.6*color[0], 0.6*color[1], 0.6*color[2])
    positions= [0.0,0.2,0.8,1.0]
    colors= [QtCore.Qt.white,col1,col2,QtCore.Qt.black]
    linear_gradient= QtGui.QLinearGradient(bounds.topLeft(), bounds.bottomRight())
    for pos,col in zip(positions, colors if not reverse else reversed(colors)):
      linear_gradient.setColorAt(pos, col)
    return linear_gradient

  def getMovableRange(self):
    if self.kind=='ellipse':
      return self.width(), self.height()
    elif self.kind=='circle':
      l= min(self.width(), self.height())
      return l,l

  def getStickRange(self):
    mw,mh= self.getMovableRange()
    stick_rad= mw*self.stick_rad
    return mw/2.0-stick_rad, mh/2.0-stick_rad

  def paintEvent(self, event):
    painter= QtGui.QPainter(self)
    painter.setPen(QtCore.Qt.SolidLine)
    painter.setRenderHint(QtGui.QPainter.Antialiasing)

    #Drawing movable range.
    mw,mh= self.getMovableRange()
    bounds= QtCore.QRectF(-mw/2+1, -mh/2+1, mw-2, mh-2  ).translated(self.center())
    painter.setBrush(self.getGradient(self.bg_color, bounds, reverse=True))
    painter.drawEllipse(bounds)

    #Drawing stick.
    mw,mh= self.getMovableRange()
    sx,sy= self.getStickRange()
    stick_displacement= self.center()+QtCore.QPointF(self.stick_pos.p2().x()*sx,-self.stick_pos.p2().y()*sy)
    stick_rad= mw*self.stick_rad
    bounds= QtCore.QRectF(-stick_rad, -stick_rad, stick_rad*2, stick_rad*2).translated(stick_displacement)
    painter.setBrush(self.getGradient(self.stick_color, bounds))
    #painter.setBrush(QtCore.Qt.black)
    painter.drawEllipse(bounds)

  def mousePressEvent(self, event):
    dp= QtCore.QLineF(self.center(), event.pos())
    mw,mh= self.getMovableRange()
    stick_rad= mw*self.stick_rad
    self.stick_grabbed= dp.length()<stick_rad
    if self.stick_grabbed:  self.pos_grabbed= event.pos()
    return super(TVirtualJoyStick, self).mousePressEvent(event)

  def mouseReleaseEvent(self, event):
    self.stick_grabbed= False
    self.resetStickPos()
    self.update()

  def mouseMoveEvent(self, event):
    if self.stick_grabbed:
      dp= QtCore.QPointF(event.pos()-self.pos_grabbed)
      sx,sy= self.getStickRange()
      self.stick_pos= QtCore.QLineF(QtCore.QPointF(0,0), QtCore.QPointF(dp.x()/sx,-dp.y()/sy))
      if self.stick_pos.length()>1.0:  self.stick_pos.setLength(1.0)
      #print 'debug', self.stick_pos, self.stick_pos.p2(), self.stick_pos.length(), self.stick_pos.angle()
      self.update()
    print(self.position())


def Print(*s):
  for ss in s:  print(ss, end=' ')
  print('')

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
    joystick1.resize(100, 100)
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

class TVirtualJoyStickTest2(QtGui.QWidget):
  def __init__(self):
    QtGui.QWidget.__init__(self)
    self.InitUI()

  def InitUI(self):
    # Set window size.
    self.resize(400, 240)

    # Set window title
    self.setWindowTitle('VirtualJoyStickTest2')

    mainlayout= QtGui.QGridLayout()

    cmbbx1= QtGui.QComboBox(self)
    cmbbx1.addItem('aaaaaa')
    cmbbx1.setCurrentIndex(0)
    self.cmbbx1= cmbbx1
    mainlayout.addWidget(self.cmbbx1,0,0,1,2)

    joystick1= TVirtualJoyStick(self)
    joystick1.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
    self.joystick1= joystick1
    mainlayout.addWidget(self.joystick1,1,0)

    btn1= QtGui.QPushButton('Disp position', self)
    btn1.clicked.connect(lambda:Print(self.joystick1.position()))
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

# Create an PyQT4 application object.
a= QtGui.QApplication(sys.argv)

# The QWidget widget is the base class of all user interface objects in PyQt4.
w= TVirtualJoyStickTest()
w2= TVirtualJoyStickTest2()

sys.exit(a.exec_())

