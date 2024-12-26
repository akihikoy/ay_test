#!/usr/bin/python
#\file    combobox4.py
#\brief   User editable combobox.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Nov.07, 2023

import sys
from PyQt4 import QtCore,QtGui

def Print(*s):
  for ss in s:  print ss,
  print ''

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
    btn1.clicked.connect(lambda:self.close() if self.cmbbx1.currentText()=='Option-2' or self.cmbbx1.currentText()=='exit' else Print('Hint: To exit, select Option-2, or type exit'))
    btn1.resize(btn1.sizeHint())
    btn1.move(100, 150)
    self.btn1= btn1

    cmbbx1= QtGui.QComboBox(self)
    cmbbx1.addItem("Option-1")
    cmbbx1.addItem("Option-2")
    cmbbx1.addItem("Option-3")
    cmbbx1.setCurrentIndex(1)
    cmbbx1.setEditable(True)
    cmbbx1.move(50, 60)
    cmbbx1.activated[str].connect(lambda: Print('1/Selected',self.cmbbx1.currentText()))
    cmbbx1.editTextChanged.connect(lambda: Print('1/TextChanged',self.cmbbx1.currentText()))
    #Print(dir(cmbbx1))
    self.cmbbx1= cmbbx1

    cmbbx2= QtGui.QComboBox(self)
    cmbbx2.addItem("Item-1")
    cmbbx2.addItem("Item-2")
    cmbbx2.addItem("Item-3")
    cmbbx2.setCurrentIndex(1)
    cmbbx2.setEditable(False)
    cmbbx2.move(180, 60)
    cmbbx2.activated[str].connect(lambda: Print('2/Selected',self.cmbbx2.currentText()))
    cmbbx2.editTextChanged.connect(lambda: Print('2/TextChanged',self.cmbbx2.currentText()))
    self.cmbbx2= cmbbx2

    # Show window
    self.show()

# Create an PyQT4 application object.
a = QtGui.QApplication(sys.argv)

# The QWidget widget is the base class of all user interface objects in PyQt4.
w = TComboBox()

sys.exit(a.exec_())
