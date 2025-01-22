#!/usr/bin/python3
#\file    checkbox01.py
#\brief   Test checkbox
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Apr.21, 2021

import sys
#from PyQt4 import QtCore,QtGui
from _import_qt import *

def Print(*s):
  for ss in s:  print(ss, end=' ')
  print('')

class TCheckBox(QtGui.QWidget):
  def __init__(self):
    QtGui.QWidget.__init__(self)
    self.InitUI()

  def InitUI(self):
    # Set window size.
    self.resize(320, 240)

    # Set window title
    self.setWindowTitle("CheckBox")

    # Add a button
    btn1= QtGui.QPushButton('_________Exit?_________', self)
    #btn1.setFlat(True)
    btn1.setToolTip('Click to make something happen')
    btn1.clicked.connect(lambda:self.close() if self.chkbx1.isChecked() else Print('Hint: Check to exit'))
    btn1.resize(btn1.sizeHint())
    btn1.move(100, 150)
    self.btn1= btn1

    chkbx1= QtGui.QCheckBox('Do you want to exit?', self)
    chkbx1.setChecked(True)
    chkbx1.move(10, 60)
    chkbx1.clicked.connect(lambda:Print('Checked!') if self.chkbx1.isChecked() else Print('Unchecked!'))
    self.chkbx1= chkbx1

    # Show window
    self.show()

# Create an PyQT4 application object.
a = QtGui.QApplication(sys.argv)

# The QWidget widget is the base class of all user interface objects in PyQt4.
w = TCheckBox()

sys.exit(a.exec_())
