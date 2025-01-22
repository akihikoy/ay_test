#!/usr/bin/python3
#\file    radiobutton3.py
#\brief   Radio button with a user-editable "Other".
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Apr.14, 2021

import sys
#from PyQt4 import QtCore,QtGui
from _import_qt import *

def Print(s):
  print(s)

class TRadioButton(QtGui.QWidget):
  def __init__(self):
    QtGui.QWidget.__init__(self)
    self.InitUI()

  def InitUI(self):
    # Set window size.
    self.resize(370, 240)

    # Set window title
    self.setWindowTitle("Radio button")

    # Add a button
    btn1= QtGui.QPushButton('_________Exit?_________', self)
    #btn1.setFlat(True)
    btn1.setToolTip('Click to make something happen')
    btn1.clicked.connect(lambda:self.close() if self.radbtn2.isChecked() or (self.radbtn3.isChecked() and self.edit1.text()=='exit') else Print('Hint: To exit, select Option-2, or Other and type exit'))
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

    radbtn3= QtGui.QRadioButton('Other', self)
    radbtn3.setCheckable(True)
    radbtn3.setFocusPolicy(QtCore.Qt.NoFocus)
    self.radbtn3= radbtn3
    radbtn3.move(190, 60)

    edit1= QtGui.QLineEdit(self)
    #edit1.setValidator(QtGui.QIntValidator())
    self.edit1= edit1
    edit1.move(270, 60)
    edit1.resize(90, 30)
    self.edit1.setEnabled(self.radbtn3.isChecked())

    group= QtGui.QButtonGroup()
    group.addButton(radbtn1,1)
    group.addButton(radbtn2,2)
    group.addButton(radbtn3,3)
    #group.addButton(edit1,4)  #NOTE: This is impossible.
    self.group= group

    def radbtn_clicked():
      self.edit1.setEnabled(self.radbtn3.isChecked())
      if isinstance(self.sender(),QtGui.QRadioButton) and self.radbtn2.isChecked():
        self.btn1.setText('Exit')
      else:
        self.btn1.setText('Not exit({text})'.format(text=self.group.checkedButton().text()))
    radbtn1.clicked.connect(radbtn_clicked)
    radbtn2.clicked.connect(radbtn_clicked)
    radbtn3.clicked.connect(radbtn_clicked)

    # Show window
    self.show()

# Create an PyQT4 application object.
a = QtGui.QApplication(sys.argv)

# The QWidget widget is the base class of all user interface objects in PyQt4.
w = TRadioButton()

sys.exit(a.exec_())
