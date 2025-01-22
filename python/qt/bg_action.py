#!/usr/bin/python3
#\file    bg_action.py
#\brief   Test if an event callback is executed in background or foreground.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Apr.16, 2021

import sys
#from PyQt4 import QtGui
from _import_qt import *

def Proc():
  print('Heavy process started...')
  x= 0
  for i in range(100000000):
    x= (x+i)/2**0.5
    if i%1000000==0:  print('  ..',i,x)
  print('  Done; x=',x)

class TButtons(QtGui.QWidget):
  def __init__(self):
    QtGui.QWidget.__init__(self)
    self.InitUI()

  def InitUI(self):
    self.resize(320, 240)
    self.setWindowTitle("Background action test")

    btn1 = QtGui.QPushButton('Heavy Process', self)
    btn1.clicked.connect(Proc)
    btn1.resize(btn1.sizeHint())
    btn1.move(100, 60)

    btn2 = QtGui.QPushButton('Light Process (1)', self)
    btn2.clicked.connect(lambda:btn2.setText('Light Process (2)' if btn2.text()=='Light Process (1)' else 'Light Process (1)'))
    btn2.resize(btn2.sizeHint())
    btn2.move(100, 140)

    # Show window
    self.show()

# Create an PyQT4 application object.
a = QtGui.QApplication(sys.argv)

# The QWidget widget is the base class of all user interface objects in PyQt4.
w = TButtons()

sys.exit(a.exec_())
