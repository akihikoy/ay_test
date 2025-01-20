#!/usr/bin/python3
#\file    label1.py
#\brief   QLabel example.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Apr.14, 2021

import sys
from PyQt4 import QtCore,QtGui

def Print(*s):
  for ss in s:  print(ss, end=' ')
  print('')

class TLabel(QtGui.QWidget):
  def __init__(self):
    QtGui.QWidget.__init__(self)
    self.InitUI()

  def InitUI(self):
    # Set window size.
    self.resize(320, 240)

    # Set window title
    self.setWindowTitle("Label")

    label1= QtGui.QLabel('Label label\nLabel', self)
    label1.move(10, 60)

    btn1= QtGui.QPushButton('Exit', self)
    #btn1.setFlat(True)
    btn1.clicked.connect(lambda:self.close())
    btn1.resize(btn1.sizeHint())
    btn1.move(100, 150)
    self.btn1= btn1

    # Show window
    self.show()

# Create an PyQT4 application object.
a = QtGui.QApplication(sys.argv)

# The QWidget widget is the base class of all user interface objects in PyQt4.
w = TLabel()

sys.exit(a.exec_())

