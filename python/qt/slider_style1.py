#!/usr/bin/python3
#\file    slider_style1.py
#\brief   Test styles of slider.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Apr.15, 2021

import sys

#from PyQt4 import QtCore,QtGui

#from PyQt5 import QtCore,QtWidgets
#QtGui= QtWidgets

from _import_qt import *

def Print(*s):
  for ss in s:  print(ss, end=' ')
  print('')

try:
  class SliderProxyStyle(QtGui.QProxyStyle):
    def pixelMetric(self, metric, option, widget):
      if metric==QtGui.QStyle.PM_SliderThickness:
        return 68
      elif metric==QtGui.QStyle.PM_SliderLength:
        return 68
      return super(SliderProxyStyle, self).pixelMetric(metric, option, widget)
except:
  SliderProxyStyle= None


class TSlider(QtGui.QWidget):
  def __init__(self):
    QtGui.QWidget.__init__(self)
    self.InitUI()

  def SetSliderStyle(self, style):
    if style=='Style-1':
      self.slider1.setStyleSheet('')
    elif style=='Style-2':
      self.slider1.setStyleSheet('''
        QSlider {
            min-height: 68px;
            max-height: 68px;
            height: 68px;
        }
        QSlider::groove:horizontal {
            background: transparent;
            border: 2px solid #333;
            height: 6px;
            padding: -0 -2px;
            margin: -0 0px;
        }
        QSlider::handle:horizontal {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #b4b4b4, stop:1 #8f8f8f);
            border: 1px solid #5c5c5c;
            width: 60px;
            margin: -30px 0;
            border-radius: 3px;
        }
        ''')
    elif style=='Style-3':
      self.slider1.setStyleSheet('''
        QSlider {
            height: 68px;
        }
        QSlider::groove:horizontal {
            background: transparent;
            border: 2px solid #aaa;
            height: 60px;
            margin: -0 0px;
        }
        QSlider::handle:horizontal {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #b4b4b4, stop:1 #8f8f8f);
            border: 1px solid #5c5c5c;
            width: 60px;
            margin: -0px 0;
            border-radius: 3px;
        }
        ''')
    elif style=='Style-4':
      self.slider1.setStyleSheet('''
        QSlider {
            height: 68px;
        }
        QSlider::groove:horizontal {
            border-radius: 1px;
            height: 3px;
            margin: 0px;
            background-color: rgb(52, 59, 72);
        }
        QSlider::handle:horizontal {
            background-color: rgb(85, 170, 255);
            border: none;
            height: 40px;
            width: 40px;
            margin: -20px 0;
            border-radius: 20px;
            padding: -20px 0px;
        }
        QSlider::handle:horizontal:hover {
            background-color: rgb(155, 180, 255);
        }
        QSlider::handle:horizontal:pressed {
            background-color: rgb(65, 255, 195);
        }
        ''')
    elif style=='Style-5':
      if SliderProxyStyle is None:
        print('Style-5 uses PyQt5. Comment out the PyQt4 import line and uncomment the PyQt import lines.')
        return
      self.slider1.setStyleSheet('')
      style= SliderProxyStyle(self.slider1.style())
      self.slider1.setStyle(style)

  def InitUI(self):
    # Set window size.
    self.resize(600, 300)

    # Set window title
    self.setWindowTitle('Slider')

    mainlayout= QtGui.QVBoxLayout()
    self.setLayout(mainlayout)

    cmbbx1= QtGui.QComboBox(self)
    cmbbx1.addItem('Style-1')
    cmbbx1.addItem('Style-2')
    cmbbx1.addItem('Style-3')
    cmbbx1.addItem('Style-4')
    cmbbx1.addItem('Style-5')
    #cmbbx1.setCurrentIndex(0)
    cmbbx1.activated[str].connect(lambda:(self.SetSliderStyle(self.cmbbx1.currentText()), Print('Selected',self.cmbbx1.currentText())))
    self.cmbbx1= cmbbx1
    mainlayout.addWidget(cmbbx1)

    vspacer0= QtGui.QSpacerItem(10, 10, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
    mainlayout.addItem(vspacer0)

    slidergroup= QtGui.QGridLayout()
    #slidergroup.move(10, 60)
    #slidergroup.resize(10, 60)
    #self.setLayout(slidergroup)
    mainlayout.addLayout(slidergroup)

    #slider1= QtGui.QSlider(QtCore.Qt.Vertical, self)
    slider1= QtGui.QSlider(QtCore.Qt.Horizontal, self)
    slider1.setTickPosition(QtGui.QSlider.TicksBothSides)
    slider1.setRange(0, 8)
    slider1.setTickInterval(1)
    slider1.setSingleStep(1)
    slider1.setValue(6)
    #slider1.move(10, 60)
    slider1.resize(100, 20)
    slider1.toValue= lambda: 1000 + 100*self.slider1.value()
    slider1.valueChanged.connect(lambda:self.label1.setText(str(self.slider1.toValue())))
    self.slider1= slider1
    slidergroup.addWidget(slider1, 0, 0, 1, 5)

    label1= QtGui.QLabel('0',self)
    self.label1= label1
    slidergroup.addWidget(label1, 0, 5, 1, 1)

    self.label1.setText(str(self.slider1.toValue()))

    labelt1= QtGui.QLabel('1000',self)
    slidergroup.addWidget(labelt1, 1, 0, 1, 1, QtCore.Qt.AlignLeft)
    labelt2= QtGui.QLabel('1200',self)
    slidergroup.addWidget(labelt2, 1, 1, 1, 1, QtCore.Qt.AlignLeft)
    labelt3= QtGui.QLabel('1400',self)
    slidergroup.addWidget(labelt3, 1, 2, 1, 1, QtCore.Qt.AlignCenter)
    labelt4= QtGui.QLabel('1600',self)
    slidergroup.addWidget(labelt4, 1, 3, 1, 1, QtCore.Qt.AlignRight)
    labelt5= QtGui.QLabel('1800',self)
    slidergroup.addWidget(labelt5, 1, 4, 1, 1, QtCore.Qt.AlignRight)

    vspacer1= QtGui.QSpacerItem(10, 10, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
    slidergroup.addItem(vspacer1, 2, 0, 1, 6)

    mainlayout.addItem(vspacer0)

    # Add a button
    btn1= QtGui.QPushButton('_________Exit?_________', self)
    #btn1.setFlat(True)
    btn1.setToolTip('Click to make something happen')
    btn1.clicked.connect(lambda:self.close() if self.slider1.toValue()<1500 else Print('Hint: Set value less than 1500 to exit'))
    btn1.resize(btn1.sizeHint())
    #btn1.move(100, 150)
    self.btn1= btn1
    mainlayout.addWidget(btn1)

    # Show window
    self.show()

# Create an PyQT4 application object.
a = QtGui.QApplication(sys.argv)

# The QWidget widget is the base class of all user interface objects in PyQt4.
w = TSlider()

sys.exit(a.exec_())
