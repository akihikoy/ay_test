#!/usr/bin/python3
# -*- coding: utf-8 -*-
#\file    grid3.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Apr.05, 2021

import sys
from PyQt4 import QtGui

def Print(s):
  print(s)

class TGrid(QtGui.QWidget):
  def __init__(self):
    QtGui.QWidget.__init__(self)
    self.InitUI()

  def InitUI(self):
    # Set window size.
    #self.resize(320, 240)

    # Set window title
    self.setWindowTitle("Grid")

    # Grid layout
    grid= QtGui.QGridLayout()
    self.setLayout(grid)

    # Add buttons on grid
    names= [[('a0\na01',lambda:Print('0')), ('a1',lambda:Print('1'))],
            [('b0',lambda:Print('x')), ('b1',lambda:Print('y')), ('b2',lambda:Print('z'))],
            [('c0',exit,2)]]
    self.buttons= []
    self.default_font_size= 18
    for r,row in enumerate(names):
      for c,contents in enumerate(row):
        name,f,colsize= contents if len(contents)==3 else (contents[0],contents[1],1)
        btn= QtGui.QPushButton(name)
        btn.clicked.connect(f)
        btn.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        btn.setFont(QtGui.QFont('', self.default_font_size))
        btn.resizeEvent= self.resizeText
        grid.addWidget(btn, r, c, 1, colsize)
        self.buttons.append(btn)

    # Show window
    self.show()

  def resizeText(self, event):
    if int(self.rect().width()/20)>self.default_font_size:
      f= QtGui.QFont('', int(self.rect().width()/20))
    else:
      f= QtGui.QFont('', self.default_font_size)
    for btn in self.buttons:  btn.setFont(f)

# Create an PyQT4 application object.
a = QtGui.QApplication(sys.argv)

# The QWidget widget is the base class of all user interface objects in PyQt4.
w = TGrid()

sys.exit(a.exec_())

