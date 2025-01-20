#!/usr/bin/python3
#\file    virtual_joy3.py
#\brief   Virtual joystick example.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Apr.30, 2022

#Ref. https://stackoverflow.com/questions/55876713/how-to-create-a-joystick-controller-widget-in-pyqt/55899694#55899694

import sys
from PyQt4 import QtCore,QtGui

class TVirtualJoyStick(QtGui.QWidget):
  onstickmoved= QtCore.pyqtSignal(list)
  def __init__(self, *args, **kwargs):
    super(TVirtualJoyStick, self).__init__(*args, **kwargs)
    #self.setMinimumSize(100, 100)
    self.setBackgroundRole(QtGui.QPalette.Base)

    self.kind= 'circle'  #circle, ellipse, hbox, vbox
    self.bg_color= [200]*3  #Color of movable range.
    self.bg_height= 0.3  #Display height of movable range (effective with kind==hbox,vbox).
    self.resetStickPos()
    self.stick_grabbed= False
    self.stick_size= 0.6  #Stick size per width/height of movable range.
    self.stick_color= [128, 128, 255]

  #kind: 2d-joy: circle, ellipse, 1d-joy: hbox, vbox
  def setKind(self, kind):
    self.kind= kind

  #Set color of movable range.
  def setBGColor(self, bg_color):
    self.bg_color= bg_color

  #Set display height of movable range (effective with kind==hbox,vbox).
  def setBGHeight(self, bg_height):
    self.bg_height= bg_height

  #Set stick size per width/height of movable range.
  def setStickSize(self, stick_size):
    self.stick_size= stick_size

  def setStickColor(self, stick_color):
    self.stick_color= stick_color

  def resetStickPos(self):
    #Normalized stick position:
    self.stick_pos= QtCore.QLineF(QtCore.QPointF(0,0),QtCore.QPointF(0,0))

  def center(self):
    return QtCore.QPointF(self.width()/2, self.height()/2)

  def minimumSizeHint(self):
    #return QtCore.QSize(100, self.heightForWidth(100))
    return QtCore.QSize(20, 20)

  #def sizeHint(self):
    #return QtCore.QSize(400, self.heightForWidth(400))

  #def heightForWidth(self, width):
    #return width*1.2

  #Return the current joystick position.
  def position(self):
    #return self.stick_pos.p2(), self.stick_pos.length(), self.stick_pos.angle()
    if self.kind in ('ellipse','circle'):
      return [self.stick_pos.p2().x(), self.stick_pos.p2().y()]
    elif self.kind=='hbox':
      return [self.stick_pos.p2().x()]
    elif self.kind=='vbox':
      return [self.stick_pos.p2().y()]

  def getGradient(self, color, bounds, reverse=False):
    col0= QtGui.QColor(min(255,1.5*color[0]), min(255,1.5*color[1]), min(255,1.5*color[2]))
    col1= QtGui.QColor(*color)
    col2= QtGui.QColor(0.6*color[0], 0.6*color[1], 0.6*color[2])
    col3= QtGui.QColor(0.2*color[0], 0.2*color[1], 0.2*color[2])
    positions= [0.0,0.2,0.8,1.0]
    #colors= [QtCore.Qt.white,col1,col2,QtCore.Qt.black]
    colors= [col0,col1,col2,col3]
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
    elif self.kind=='hbox':
      return self.width(), self.width()*self.bg_height
    elif self.kind=='vbox':
      return self.height()*self.bg_height, self.height()

  def getStickSize(self):
    mw,mh= self.getMovableRange()
    if self.kind in ('ellipse','circle'):
      return mw*self.stick_size, mw*self.stick_size
    elif self.kind=='hbox':
      return mw*self.stick_size, mh
    elif self.kind=='vbox':
      return mw, mh*self.stick_size

  def getStickRange(self):
    mw,mh= self.getMovableRange()
    sw,sh= self.getStickSize()
    if self.kind in ('ellipse','circle'):
      return mw/2.0-sw/2.0, mh/2.0-sh/2.0
    elif self.kind=='hbox':
      return mw/2.0-sw/2.0, 0
    elif self.kind=='vbox':
      return 0, mh/2.0-sh/2.0

  def paintEvent(self, event):
    painter= QtGui.QPainter(self)
    painter.setPen(QtCore.Qt.SolidLine)
    painter.setRenderHint(QtGui.QPainter.Antialiasing)

    #Drawing movable range.
    mw,mh= self.getMovableRange()
    bounds= QtCore.QRectF(-mw/2+1, -mh/2+1, mw-2, mh-2).translated(self.center())
    painter.setBrush(self.getGradient(self.bg_color, bounds, reverse=True))
    if self.kind in ('ellipse','circle'):
      painter.drawEllipse(bounds)
    elif self.kind in ('hbox','vbox'):
      painter.drawRect(bounds)

    #Drawing stick.
    sx,sy= self.getStickRange()
    stick_displacement= self.center()+QtCore.QPointF(self.stick_pos.p2().x()*sx,-self.stick_pos.p2().y()*sy)
    sw,sh= self.getStickSize()
    bounds= QtCore.QRectF(-sw/2.0+1, -sh/2.0+1, sw-2, sh-2).translated(stick_displacement)
    painter.setBrush(self.getGradient(self.stick_color, bounds))
    #painter.setBrush(QtCore.Qt.black)
    if self.kind in ('ellipse','circle'):
      painter.drawEllipse(bounds)
    elif self.kind in ('hbox','vbox'):
      painter.drawRect(bounds)

  def mousePressEvent(self, event):
    dp= QtCore.QLineF(self.center(), event.pos())
    sw,sh= self.getStickSize()
    if self.kind in ('ellipse','circle'):
      self.stick_grabbed= dp.length()<sw
    elif self.kind in ('hbox','vbox'):
      self.stick_grabbed= abs(dp.dx())<sw/2.0 and abs(dp.dy())<sh/2.0
    if self.stick_grabbed:  self.pos_grabbed= event.pos()
    self.onstickmoved.emit(self.position())
    return super(TVirtualJoyStick, self).mousePressEvent(event)

  def mouseReleaseEvent(self, event):
    self.stick_grabbed= False
    self.resetStickPos()
    self.update()
    self.onstickmoved.emit(self.position())

  def mouseMoveEvent(self, event):
    if self.stick_grabbed:
      dp= QtCore.QPointF(event.pos()-self.pos_grabbed)
      sx,sy= self.getStickRange()
      self.stick_pos= QtCore.QLineF(QtCore.QPointF(0,0), QtCore.QPointF(dp.x()/sx if sx>0 else 0,-dp.y()/sy if sy>0 else 0))
      if self.stick_pos.length()>1.0:  self.stick_pos.setLength(1.0)
      self.update()
      self.onstickmoved.emit(self.position())
    #print self.position()


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
    joystick1.move(10, 10)
    joystick1.resize(100, 100)
    self.joystick1= joystick1
    #mainlayout.addWidget(joystick1)

    joystick2= TVirtualJoyStick(self)
    joystick2.setKind('hbox')
    joystick2.move(10, 120)
    joystick2.resize(100, 40)
    self.joystick2= joystick2

    joystick3= TVirtualJoyStick(self)
    joystick3.setKind('vbox')
    joystick3.move(120, 10)
    joystick3.resize(40, 100)
    self.joystick3= joystick3

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

    joystick2= TVirtualJoyStick(self)
    joystick2.setKind('hbox')
    joystick2.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
    joystick2.onstickmoved.connect(lambda p:self.btn2.setText('Exit({})'.format(p)))
    self.joystick2= joystick2
    mainlayout.addWidget(self.joystick2,0,0,1,2)

    joystick1= TVirtualJoyStick(self)
    joystick1.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
    joystick1.onstickmoved.connect(lambda p:self.btn2.setText('Exit({})'.format(p)))
    self.joystick1= joystick1
    mainlayout.addWidget(self.joystick1,1,0)

    btn1= QtGui.QPushButton('Disp position', self)
    btn1.clicked.connect(lambda:Print(self.joystick1.position()))
    btn1.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
    self.btn1= btn1
    mainlayout.addWidget(self.btn1,1,1)

    btn2= QtGui.QPushButton('Exit', self)
    btn2.clicked.connect(lambda:self.close())
    btn2.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Fixed)
    self.btn2= btn2
    mainlayout.addWidget(self.btn2,2,0,1,2)

    joystick3= TVirtualJoyStick(self)
    joystick3.setKind('vbox')
    joystick3.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
    joystick3.onstickmoved.connect(lambda p:self.btn2.setText('Exit({})'.format(p)))
    self.joystick3= joystick3
    mainlayout.addWidget(self.joystick3,0,2,3,1)

    self.setLayout(mainlayout)

    # Show window
    self.show()

# Create an PyQT4 application object.
a= QtGui.QApplication(sys.argv)

# The QWidget widget is the base class of all user interface objects in PyQt4.
w= TVirtualJoyStickTest()
w2= TVirtualJoyStickTest2()

sys.exit(a.exec_())

