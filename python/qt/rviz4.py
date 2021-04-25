#!/usr/bin/python
#\file    rviz4.py
#\brief   Embedding RViz as a Qt widget (delayed configuration);
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


class TRVizUtil(rviz.VisualizationFrame):
  def __init__(self):
    print '##DEBUG-1.0'
    super(TRVizUtil, self).__init__()
    print '##DEBUG-1.1'

    self.reader= rviz.YamlConfigReader()
    self.config= rviz.Config()
    self.setSplashPath('')
    self.setMenuBar(None)
    self.setStatusBar(None)
    self.is_initialized= False

    self.config_file= 'rviz1_config.rviz'

  def Initialize(self):
    if not self.is_initialized:
      self.initialize()
      self.is_initialized= True
    self.reader.readFile(self.config, self.config_file)
    self.load(self.config)
    self.setMenuBar(None)
    self.setStatusBar(None)
    self.setHideButtonVisibility(False)


class TRVizExample(QtGui.QWidget):
  def __init__(self):
    QtGui.QWidget.__init__(self)
    self.close_callback= None
    self.InitUI()

  def InitUI(self):
    # Set window size.
    self.resize(320, 240)

    # Set window title
    self.setWindowTitle('RVizExample')

    mainlayout= QtGui.QGridLayout()

    print '##DEBUG-1'
    self.rviz_widget= TRVizUtil()
    print '##DEBUG-2'
    global p_roscore
    if p_roscore is not None:
      self.rviz_widget.Initialize()
    print '##DEBUG-3'
    mainlayout.addWidget(self.rviz_widget,0,0)

    btn1= QtGui.QPushButton('InitRViz', self)
    btn1.clicked.connect(lambda:(self.rviz_widget.Initialize()  ))
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

  # Override closing event
  def closeEvent(self, event):
    if self.close_callback is not None:
      res= self.close_callback(event)
      if res in (None,True):
        event.accept()
      else:
        event.ignore()
    else:
      event.accept()


# Create an PyQT4 application object.
a = QtGui.QApplication(sys.argv)

p_roscore= None

# The QWidget widget is the base class of all user interface objects in PyQt4.
w = TRVizExample()

import subprocess
p_roscore= subprocess.Popen('roscore')
w.close_callback= lambda event: p_roscore.terminate()

sys.exit(a.exec_())


