#!/usr/bin/python3
#\file    toolbox1.py
#\brief   Test of QToolBox.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.19, 2023

import sys
#from PyQt4 import QtCore,QtGui
from _import_qt import *

def Print(*s):
  for ss in s:  print(ss, end=' ')
  print('')

class TToolBox(QtGui.QWidget):
  def __init__(self):
    QtGui.QWidget.__init__(self)
    self.InitUI()

  def InitUI(self):
    # Set window size.
    self.resize(320, 240)

    # Set window title
    self.setWindowTitle("ToolBox")

    layout= QtGui.QVBoxLayout()
    layout.setContentsMargins(0, 0, 0, 0)

    toolbox= QtGui.QToolBox()

    # Create tabs
    tab1= QtGui.QWidget()
    tab2= QtGui.QWidget()
    tab3= QtGui.QWidget()
    tab4= QtGui.QWidget()

    # Add tabs
    toolbox.addItem(tab1,"Tab 1")
    toolbox.addItem(tab2,"Tab 2")
    toolbox.addItem(tab3,"Tab 3")
    toolbox.addItem(tab4,"Tab 4")

    # Resize width and height
    toolbox.resize(250, 150)

    # Set layout of first tab
    vBoxlayout  = QtGui.QVBoxLayout()
    pushButton1 = QtGui.QPushButton("Start")
    pushButton2 = QtGui.QPushButton("Settings")
    pushButton3 = QtGui.QPushButton("Stop")
    vBoxlayout.addWidget(pushButton1)
    vBoxlayout.addWidget(pushButton2)
    vBoxlayout.addWidget(pushButton3)
    tab1.setLayout(vBoxlayout)

    layout.addWidget(toolbox)

    # Set current tab to tab2
    toolbox.setCurrentWidget(tab2)

    self.setLayout(layout)

    # Show window
    self.show()

# Create an PyQT4 application object.
a = QtGui.QApplication(sys.argv)

# The QWidget widget is the base class of all user interface objects in PyQt4.
w = TToolBox()

sys.exit(a.exec_())

