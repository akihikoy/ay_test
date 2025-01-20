#!/usr/bin/python3
#\file    layout2.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Mar.31, 2017

import sys
from PyQt4 import QtGui

def Print(s):
  print(s)

class TLayout2(QtGui.QWidget):
  def __init__(self):
    QtGui.QWidget.__init__(self)
    self.InitUI()

  def InitUI(self):
    # Set window size.
    #self.resize(320, 240)

    # Set window title
    self.setWindowTitle("Layout in layout")

    # Horizontal box layout
    hBoxlayout= QtGui.QHBoxLayout()
    self.setLayout(hBoxlayout)

    btn1= QtGui.QPushButton("Exit")
    btn1.clicked.connect(exit)
    hBoxlayout.addWidget(btn1)

    # Grid layout
    grid= QtGui.QGridLayout()
    wg= QtGui.QWidget()
    wg.setLayout(grid)
    hBoxlayout.addWidget(wg)

    # Add buttons on grid
    names= [[('a0',lambda:Print('0')), ('a1',lambda:Print('1'))],
            [('b0',lambda:Print('x')), ('b1',lambda:Print('y')), ('b2',lambda:Print('z'))],
            [('c0',exit)]]
    for r,row in enumerate(names):
      for c,(name,f) in enumerate(row):
        btn= QtGui.QPushButton(name)
        btn.clicked.connect(f)
        grid.addWidget(btn, r, c)

    # Show window
    self.show()

# Create an PyQT4 application object.
a = QtGui.QApplication(sys.argv)

# The QWidget widget is the base class of all user interface objects in PyQt4.
w = TLayout2()

sys.exit(a.exec_())
