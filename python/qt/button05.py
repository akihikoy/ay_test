#!/usr/bin/python3
#\file    button05.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Apr.26, 2021

import sys
#from PyQt4 import QtCore, QtGui
from _import_qt import *

def Print(s):
  print(s)

class TButton(QtGui.QWidget):
  def __init__(self):
    QtGui.QWidget.__init__(self)
    self.InitUI()

  def InitUI(self):
    # Set window size.
    self.resize(320, 240)

    # Set window title
    self.setWindowTitle("Checkable button")

    # Add a button
    btn1= QtGui.QPushButton('Enable', self)
    #btn1.setFlat(True)
    btn1.setCheckable(True)
    btn1.setFocusPolicy(QtCore.Qt.NoFocus)
    btn1.setToolTip('Click to activate Exit button')
    btn1.clicked.connect(lambda b,btn1=btn1: Print('Enabled') if btn1.isChecked() else Print('Disabled'))
    btn1.toggled.connect(lambda checked,btn1=btn1: btn1.setText('Disable') if checked else btn1.setText('Enable'))
    #btn1.resize(btn1.sizeHint())
    btn1.move(100, 60)

    btn1.setSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
    #btn1.setMinimumWidth(btn1.minimumSizeHint().width()*0.5)
    #btn1.resize(btn1.minimumSizeHint().width(), btn1.sizeHint().height())
    #btn1.setContentsMargins(0, 0, 0, 0)
    btn1.setStyleSheet('padding:5px 10px 5px 10px')

    btn2= QtGui.QPushButton('Exit', self)
    btn2.setToolTip('Click to exit')
    btn2.resize(btn2.sizeHint())
    btn2.clicked.connect(lambda b,btn1=btn1: self.close() if btn1.isChecked() else Print('Activation button is not checked'))
    btn2.move(100, 150)

    # Show window
    self.show()

# Create an PyQT4 application object.
a = QtGui.QApplication(sys.argv)

# The QWidget widget is the base class of all user interface objects in PyQt4.
w = TButton()

sys.exit(a.exec_())
