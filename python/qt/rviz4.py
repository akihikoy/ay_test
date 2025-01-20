#!/usr/bin/python3
#\file    rviz4.py
#\brief   Embedding RViz as a Qt widget (delayed configuration);
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Apr.25, 2021

import os,sys
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
    print('##DEBUG-1.0')
    super(TRVizUtil, self).__init__()
    print('##DEBUG-1.1')

    self.reader= rviz.YamlConfigReader()
    self.config= rviz.Config()
    self.setSplashPath('')
    self.setMenuBar(None)
    self.setStatusBar(None)
    self.is_initialized= False

    self.config_file= 'rviz1_config.rviz'
    #self.config_file= os.environ['HOME']+'/.rviz/default.rviz'

  def Initialize(self):
    if not self.is_initialized:
      self.initialize()
      self.is_initialized= True
      self.reader.readFile(self.config, self.config_file)
      #print self.config.mapGetChild('Window Geometry').mapGetChild('Displays').mapGetChild('collapsed').setValue(u'true')
      #print self.config.mapGetChild('Window Geometry').mapGetChild('Displays').mapGetChild('collapsed').getValue()
      #self.config.mapGetChild('Window Geometry').mapGetChild('QMainWindow State').setValue(u'000000ff00000000fd00000004000000000000016a0000029ffc020000000afb0000001200530065006c0065006300740069006f006e00000001e10000009b0000006f00fffffffb0000001e0054006f006f006c002000500072006f007000650072007400690065007302000001ed000001df00000185000000a3fb000000120056006900650077007300200054006f006f02000001df000002110000018500000122fb000000200054006f006f006c002000500072006f0070006500720074006900650073003203000002880000011d000002210000017afb000000100044006900730070006c00610079007300000000430000029f000000f300fffffffb0000002000730065006c0065006300740069006f006e00200062007500660066006500720200000138000000aa0000023a00000294fb00000014005700690064006500530074006500720065006f02000000e6000000d2000003ee0000030bfb0000000c004b0069006e0065006300740200000186000001060000030c00000261fb0000000a0049006d006100670065000000019e0000017b0000000000000000fb0000000a0049006d006100670065000000013f000001060000000000000000000000010000010f0000028dfc0200000003fb0000001e0054006f006f006c002000500072006f00700065007200740069006500730100000041000000780000000000000000fb0000000a0056006900650077007300000000350000028d000000bd00fffffffb0000001200530065006c0065006300740069006f006e010000025a000000b200000000000000000000000200000490000000a9fc0100000001fb0000000a00560069006500770073030000004e00000080000002e10000019700000003000004b00000003efc0100000002fb0000000800540069006d00650000000000000004b0000002ec00fffffffb0000000800540069006d00650100000000000004500000000000000000000004e2000002c700000004000000040000000800000008fc0000000100000002000000010000000a0054006f006f006c00730000000000ffffffff0000000000000000')
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

    print('##DEBUG-1')
    self.rviz_widget= TRVizUtil()
    print('##DEBUG-2')
    global p_roscore
    if p_roscore is not None:
      self.rviz_widget.Initialize()
    print('##DEBUG-3')
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


