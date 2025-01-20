#!/usr/bin/python3
#\file    tabs2.py
#\brief   Tab test 2.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Apr.15, 2021

import sys
from PyQt4 import QtCore,QtGui

def Print(*s):
  for ss in s:  print(ss, end=' ')
  print('')

class TTab(QtGui.QWidget):
  def __init__(self):
    QtGui.QWidget.__init__(self)
    self.InitUI()

  def InitUI(self):
    # Set window size.
    self.resize(320, 240)

    # Set window title
    self.setWindowTitle("Tab")

    layout= QtGui.QVBoxLayout()
    layout.setContentsMargins(0, 0, 0, 0)

    tabs= QtGui.QTabWidget()

    # Create tabs
    tab1= QtGui.QWidget()
    tab2= QtGui.QWidget()
    tab3= QtGui.QWidget()
    tab4= QtGui.QWidget()

    # Add tabs
    tabs.addTab(tab1,"Tab 1")
    tabs.addTab(tab2,"Tab 2")
    tabs.addTab(tab3,"Tab 3")
    tabs.addTab(tab4,"Tab 4")

    # Resize width and height
    tabs.resize(250, 150)

    # Set layout of first tab
    vBoxlayout  = QtGui.QVBoxLayout()
    pushButton1 = QtGui.QPushButton("Start")
    pushButton2 = QtGui.QPushButton("Settings")
    pushButton3 = QtGui.QPushButton("Stop")
    vBoxlayout.addWidget(pushButton1)
    vBoxlayout.addWidget(pushButton2)
    vBoxlayout.addWidget(pushButton3)
    tab1.setLayout(vBoxlayout)

    layout.addWidget(tabs)

    # Set current tab to tab2
    tabs.setCurrentWidget(tab2)

    self.setLayout(layout)

    # Show window
    self.show()

# Create an PyQT4 application object.
a = QtGui.QApplication(sys.argv)

# The QWidget widget is the base class of all user interface objects in PyQt4.
w = TTab()

sys.exit(a.exec_())

