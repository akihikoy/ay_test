#!/usr/bin/python
#\file    radiobutton1.py
#\brief   Test radio button.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Sep.02, 2018

import sys
from PyQt4 import QtCore,QtGui

def Print(s):
  print s

class TRadioButton(QtGui.QWidget):
  def __init__(self):
    QtGui.QWidget.__init__(self)
    self.InitUI()

  def InitUI(self):
    # Set window size.
    self.resize(320, 240)

    # Set window title
    self.setWindowTitle("Radio button")

    # Add a button
    btn1= QtGui.QPushButton('_________Exit?_________', self)
    #btn1.setFlat(True)
    btn1.setToolTip('Click to make something happen')
    btn1.clicked.connect(lambda:self.close() if self.radbtn2.isChecked() else Print('Hint: Select Option-2 to exit'))
    btn1.resize(btn1.sizeHint())
    btn1.move(100, 150)
    self.btn1= btn1

    radbtn1= QtGui.QRadioButton('Option-1', self)
    radbtn1.setCheckable(True)
    radbtn1.setFocusPolicy(QtCore.Qt.NoFocus)
    radbtn1.move(10, 60)
    radbtn1.setChecked(True)

    radbtn2= QtGui.QRadioButton('Option-2', self)
    radbtn2.setCheckable(True)
    radbtn2.setFocusPolicy(QtCore.Qt.NoFocus)
    self.radbtn2= radbtn2
    radbtn2.move(100, 60)

    radbtn3= QtGui.QRadioButton('Option-3', self)
    radbtn3.setCheckable(True)
    radbtn3.setFocusPolicy(QtCore.Qt.NoFocus)
    radbtn3.move(190, 60)

    group= QtGui.QButtonGroup()
    group.addButton(radbtn1,1)
    group.addButton(radbtn2,2)
    group.addButton(radbtn3,3)

    clicked= lambda: self.btn1.setText('Exit') if isinstance(self.sender(),QtGui.QRadioButton) and self.radbtn2.isChecked() else self.btn1.setText('Not exit({text})'.format(text=group.checkedButton().text()))
    radbtn1.clicked.connect(clicked)
    radbtn2.clicked.connect(clicked)
    radbtn3.clicked.connect(clicked)

    # Show window
    self.show()

# Create an PyQT4 application object.
a = QtGui.QApplication(sys.argv)

# The QWidget widget is the base class of all user interface objects in PyQt4.
w = TRadioButton()

sys.exit(a.exec_())
