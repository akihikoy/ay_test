#!/usr/bin/python
#\file    rviz2.py
#\brief   Embedding RViz as a Qt widget (simplified example);
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Apr.25, 2021

import sys
import roslib
roslib.load_manifest('rviz')

#from PyQt4 import QtCore,QtGui

#Quick solution to use PyQt4 program with PyQt5.
from PyQt5 import QtCore,QtWidgets
import PyQt5.QtGui as PyQt5QtGui
QtGui= QtWidgets
for component in ('QFont', 'QPalette', 'QColor', 'QLinearGradient', 'QPainter'):
  setattr(QtGui,component, getattr(PyQt5QtGui,component))

#from python_qt_binding.QtGui import *
#from python_qt_binding.QtCore import *
import rviz


class TRVizExample(QtGui.QWidget):
  def __init__(self):
    QtGui.QWidget.__init__(self)
    self.InitUI()

  def InitUI(self):
    # Set window size.
    self.resize(320, 240)

    # Set window title
    self.setWindowTitle('RVizExample')

    mainlayout= QtGui.QGridLayout()

    rviz_widget= rviz.VisualizationFrame()
    rviz_widget.setSplashPath('')
    rviz_widget.initialize()
    reader= rviz.YamlConfigReader()
    config= rviz.Config()
    reader.readFile(config, 'rviz1_config.rviz')
    rviz_widget.load(config)
    rviz_widget.setMenuBar(None)
    rviz_widget.setStatusBar(None)
    rviz_widget.setHideButtonVisibility(False)
    self.rviz_widget= rviz_widget
    mainlayout.addWidget(self.rviz_widget,0,0)

    btn1= QtGui.QPushButton('Button1', self)
    btn1.clicked.connect(lambda:None)
    btn1.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
    self.btn1= btn1
    mainlayout.addWidget(self.btn1,0,1)

    btn2= QtGui.QPushButton('Exit', self)
    btn2.clicked.connect(lambda:self.close())
    btn2.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
    self.btn2= btn2
    mainlayout.addWidget(self.btn2,2,0,1,2)

    self.setLayout(mainlayout)

    # Show window
    self.show()



# Create an PyQT4 application object.
a = QtGui.QApplication(sys.argv)

# The QWidget widget is the base class of all user interface objects in PyQt4.
w = TRVizExample()

sys.exit(a.exec_())


