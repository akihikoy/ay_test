#!/usr/bin/python
#\file    button03.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Dec.14, 2016

#Class version of button01.py

import sys
from PyQt4 import QtGui

class TButton(QtGui.QWidget):
  def __init__(self):
    QtGui.QWidget.__init__(self)
    self.InitUI()

  def InitUI(self):
    # Set window size.
    self.resize(320, 240)

    # Set window title
    self.setWindowTitle("Hello World!")

    # Add a button
    btn = QtGui.QPushButton('Hello World!', self)
    btn.setToolTip('Click to quit!')
    btn.clicked.connect(exit)
    btn.resize(btn.sizeHint())
    btn.move(100, 80)

    # Show window
    self.show()

# Create an PyQT4 application object.
a = QtGui.QApplication(sys.argv)

# The QWidget widget is the base class of all user interface objects in PyQt4.
w = TButton()

sys.exit(a.exec_())
