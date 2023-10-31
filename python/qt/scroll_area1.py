#!/usr/bin/python
#\file    scroll_area1.py
#\brief   Test of QScrollArea.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Oct.31, 2023

#ref. https://stackoverflow.com/questions/46024724/pyqt-how-to-create-a-scrollable-window

import sys
from PyQt4 import QtCore, QtGui

def Print(s):
  print s

class TButton(QtGui.QWidget):
  def __init__(self):
    QtGui.QWidget.__init__(self)
    self.InitUI()

  def InitUI(self):
    # Set window size.
    self.resize(320, 240)

    # Set window title
    self.setWindowTitle("Scroll area")

    hbox= QtGui.QHBoxLayout()

    #Left panel (only one button):
    vbox1= QtGui.QVBoxLayout()

    btn2= QtGui.QPushButton('Exit', self)
    btn2.setToolTip('Click to exit')
    btn2.clicked.connect(lambda b: self.close())
    #btn2.setSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
    #btn2.resize(120,120)
    vbox1.addWidget(btn2)

    hbox.addLayout(vbox1)

    #Right panel (many buttons):
    vbox2= QtGui.QVBoxLayout()
    # Add buttons
    for i in range(10):
      btn1= QtGui.QPushButton('Button-{}'.format(i), self)
      #btn1.setFlat(True)
      btn1.clicked.connect(lambda b,i=i: Print('Button-{} pressed'.format(i)))
      #btn1.setSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
      #btn1.resize(120,120)
      vbox2.addWidget(btn1)

    #Scroll Area Properties
    widget= QtGui.QWidget()
    widget.setLayout(vbox2)
    scroll= QtGui.QScrollArea()  # Scroll Area which contains the widgets, set as the centralWidget
    scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
    scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
    scroll.setWidgetResizable(True)
    scroll.setWidget(widget)

    vbox3= QtGui.QVBoxLayout()
    vbox3.addWidget(scroll)

    hbox.addLayout(vbox3)

    self.setLayout(hbox)

    # Show window
    self.show()

# Create an PyQT4 application object.
a = QtGui.QApplication(sys.argv)

# The QWidget widget is the base class of all user interface objects in PyQt4.
w = TButton()

sys.exit(a.exec_())
