#!/usr/bin/python3
#\file    tabs4.py
#\brief   Test tab 4: Multiple tab lines.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.19, 2023

import sys
#from PyQt4 import QtCore,QtGui
#from PyQt5 import QtCore,QtWidgets
#import PyQt5.QtGui as PyQt5QtGui
#QtGui= QtWidgets
#QtGui.QPainter= PyQt5QtGui.QPainter
from _import_qt import *

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

    '''
    NOTE:
    QTabWidget does not provide multiple rows of tab bars.
    We create three QTabWidget to make two rows of tab bars.
    One QTabWidget has the tab bar at side that has two tabs,
    and the other two QTabWidget are placed in each tab of the side QTabWidget.
    '''

    tabs_side= QtGui.QTabWidget()

    # This rotates the tabs 90 degrees so they run vertically on the side.
    tabs_side.setTabPosition(QtGui.QTabWidget.West)
    #tabs_side.setTabShape(QtGui.QTabWidget.Rounded)

    # This sets the tab bar's style to resemble a list rather than tabs,
    # which allows multiple rows of tabs when the tabs are rotated.
    #tabs.tabBar().setDocumentMode(True)

    tabs_top1= QtGui.QTabWidget()
    tabs_top2= QtGui.QTabWidget()

    # Create side tabs
    tab_side1= QtGui.QWidget()
    tab_side2= QtGui.QWidget()

    # Add side tabs
    tabs_side.addTab(tab_side1,"Side 1")
    tabs_side.addTab(tab_side2,"Side 2")

    # Create layouts for side tabs
    layout_side1= QtGui.QVBoxLayout()
    layout_side1.setContentsMargins(0, 0, 0, 0)
    layout_side1.addWidget(tabs_top1)
    layout_side2= QtGui.QVBoxLayout()
    layout_side2.setContentsMargins(0, 0, 0, 0)
    layout_side2.addWidget(tabs_top2)
    tab_side1.setLayout(layout_side1)
    tab_side2.setLayout(layout_side2)

    # Create top tabs
    tab_top1= QtGui.QWidget()
    tab_top2= QtGui.QWidget()
    tab_top3= QtGui.QWidget()
    tab_top4= QtGui.QWidget()

    # Add top tabs
    tabs_top1.addTab(tab_top1,"Tab 1")
    tabs_top1.addTab(tab_top2,"Tab 2")
    tabs_top2.addTab(tab_top3,"Tab 3")
    tabs_top2.addTab(tab_top4,"Tab 4")

    # Resize width and height
    tabs_side.resize(250, 150)

    # Set layout of first tab
    vBoxlayout  = QtGui.QVBoxLayout()
    pushButton1 = QtGui.QPushButton("Start")
    pushButton2 = QtGui.QPushButton("Settings")
    pushButton3 = QtGui.QPushButton("Stop")
    vBoxlayout.addWidget(pushButton1)
    vBoxlayout.addWidget(pushButton2)
    vBoxlayout.addWidget(pushButton3)
    tab_top1.setLayout(vBoxlayout)

    layout.addWidget(tabs_side)

    # Set current tab to tab2
    tabs_side.setCurrentWidget(tab_side1)
    tabs_top1.setCurrentWidget(tab_top2)

    self.setLayout(layout)

    # Show window
    self.show()

# Create an PyQT4 application object.
a = QtGui.QApplication(sys.argv)

# The QWidget widget is the base class of all user interface objects in PyQt4.
w = TTab()

sys.exit(a.exec_())

