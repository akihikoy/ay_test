#!/usr/bin/python
#\file    lineedit2.py
#\brief   Test of expanding one line text box;
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Apr.14, 2021

import sys
from PyQt4 import QtCore,QtGui

def Print(*s):
  for ss in s:  print ss,
  print ''

class TComboBox(QtGui.QWidget):
  def __init__(self):
    QtGui.QWidget.__init__(self)
    self.InitUI()

  def InitUI(self):
    # Set window size.
    self.resize(320, 240)

    # Set window title
    self.setWindowTitle("LineEdit")

    mainlayout= QtGui.QVBoxLayout()
    #mainlayout= QtGui.QGridLayout()
    self.setLayout(mainlayout)

    edit1= QtGui.QLineEdit(self)
    edit1.setValidator(QtGui.QIntValidator())
    edit1.move(10, 60)
    edit1.font_size= (10,30)
    edit1.setFont(QtGui.QFont('', edit1.font_size[0]))
    edit1.resizeEvent= lambda event,obj=edit1: self.ResizeText(obj,event)
    self.edit1= edit1
    mainlayout.addWidget(edit1)#,0,1)

    sublayout1= QtGui.QHBoxLayout()
    mainlayout.addLayout(sublayout1)

    hspacer1= QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
    sublayout1.addSpacerItem(hspacer1)#,0,1)

    # Add a button
    btn1= QtGui.QPushButton('__Exit?__', self)
    #btn1.setFlat(True)
    btn1.setToolTip('Click to make something happen')
    btn1.clicked.connect(lambda:self.close() if int(self.edit1.text())%2==0 else Print('Hint: Set an even number to exit'))
    btn1.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
    btn1.resize(btn1.sizeHint())
    btn1.move(100, 150)
    btn1.font_size= (10,30)
    btn1.setFont(QtGui.QFont('', btn1.font_size[0]))
    btn1.resizeEvent= lambda event,obj=btn1: self.ResizeText(obj,event)
    self.btn1= btn1
    sublayout1.addWidget(btn1)#,0,1)

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
w = TComboBox()

sys.exit(a.exec_())

