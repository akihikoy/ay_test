#!/usr/bin/python3
#\file    combobox2.py
#\brief   Test expanding ComboBox
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Apr.14, 2021

import sys
#from PyQt4 import QtCore,QtGui
from _import_qt import *

def Print(*s):
  for ss in s:  print(ss, end=' ')
  print('')

class TComboBox(QtGui.QWidget):
  def __init__(self):
    QtGui.QWidget.__init__(self)
    self.InitUI()

  def InitUI(self):
    # Set window size.
    self.resize(320, 240)

    # Set window title
    self.setWindowTitle("ComboBox")

    mainlayout= QtGui.QVBoxLayout()
    #mainlayout= QtGui.QGridLayout()
    self.setLayout(mainlayout)

    cmbbx1= QtGui.QComboBox(self)
    cmbbx1.addItem("Option-1")
    cmbbx1.addItem("Option-2")
    cmbbx1.addItem("Option-3")
    cmbbx1.setCurrentIndex(1)
    cmbbx1.setFocusPolicy(QtCore.Qt.NoFocus)
    cmbbx1.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
    cmbbx1.move(10, 60)
    cmbbx1.font_size= (10,30)
    cmbbx1.setFont(QtGui.QFont('', cmbbx1.font_size[0]))
    cmbbx1.resizeEvent= lambda event,obj=cmbbx1: self.ResizeText(obj,event)
    cmbbx1.activated[str].connect(lambda:Print('Selected',self.cmbbx1.currentText()))
    self.cmbbx1= cmbbx1
    mainlayout.addWidget(cmbbx1)#,0,1)

    sublayout1= QtGui.QHBoxLayout()
    mainlayout.addLayout(sublayout1)

    vspacer1= QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
    sublayout1.addSpacerItem(vspacer1)#,0,1)
    #sublayout1.addStretch()

    # Add a button
    btn1= QtGui.QPushButton('__Exit?__', self)
    #btn1.setFlat(True)
    btn1.setToolTip('Click to make something happen')
    btn1.clicked.connect(lambda:self.close() if self.cmbbx1.currentText()=='Option-2' else Print('Hint: Select Option-2 to exit'))
    #btn1.resize(btn1.sizeHint())
    btn1.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
    btn1.move(100, 150)
    btn1.font_size= (10,30)
    btn1.setFont(QtGui.QFont('', btn1.font_size[0]))
    btn1.resizeEvent= lambda event,obj=btn1: self.ResizeText(obj,event)
    self.btn1= btn1
    sublayout1.addWidget(btn1)#,0,0)

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
