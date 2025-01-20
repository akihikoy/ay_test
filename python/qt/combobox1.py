#!/usr/bin/python3
#\file    combobox1.py
#\brief   Test ComboBox
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    May.30, 2019

import sys
from PyQt4 import QtCore,QtGui

def Print(*s):
  for ss in s:  print(ss, end=' ')
  print('')

class TComboBox(QtGui.QWidget):
  def __init__(self):
    QtGui.QWidget.__init__(self)
    self.InitUI()

  def InitUI(self):
    # Set window size.
    self.resize(320, 240)

    # Set window title
    self.setWindowTitle("ComboBox")

    # Add a button
    btn1= QtGui.QPushButton('_________Exit?_________', self)
    #btn1.setFlat(True)
    btn1.setToolTip('Click to make something happen')
    btn1.clicked.connect(lambda:self.close() if self.cmbbx1.currentText()=='Option-2' else Print('Hint: Select Option-2 to exit'))
    btn1.resize(btn1.sizeHint())
    btn1.move(100, 150)
    self.btn1= btn1

    cmbbx1= QtGui.QComboBox(self)
    cmbbx1.addItem("Option-1")
    cmbbx1.addItem("Option-2")
    cmbbx1.addItem("Option-3")
    cmbbx1.setCurrentIndex(1)
    cmbbx1.move(10, 60)
    cmbbx1.activated[str].connect(lambda:Print('Selected',self.cmbbx1.currentText()))
    self.cmbbx1= cmbbx1

    # Show window
    self.show()

# Create an PyQT4 application object.
a = QtGui.QApplication(sys.argv)

# The QWidget widget is the base class of all user interface objects in PyQt4.
w = TComboBox()

sys.exit(a.exec_())
