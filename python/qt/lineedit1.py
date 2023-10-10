#!/usr/bin/python
#\file    lineedit1.py
#\brief   Test of one line text box;
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Apr.14, 2021

import sys
from PyQt4 import QtCore,QtGui

def Print(*s):
  for ss in s:  print ss,
  print ''

class TLineEdit(QtGui.QWidget):
  def __init__(self):
    QtGui.QWidget.__init__(self)
    self.InitUI()

  def InitUI(self):
    # Set window size.
    self.resize(320, 240)

    # Set window title
    self.setWindowTitle("LineEdit")

    # Add a button
    btn1= QtGui.QPushButton('_________Exit?_________', self)
    #btn1.setFlat(True)
    btn1.setToolTip('Click to make something happen')
    btn1.clicked.connect(lambda:self.close() if self.edit1.text()!='' and int(self.edit1.text())%2==0 else Print('Hint: Put an even number to exit'))
    btn1.resize(btn1.sizeHint())
    btn1.move(100, 150)
    self.btn1= btn1

    edit1= QtGui.QLineEdit(self)
    edit1.setValidator(QtGui.QIntValidator())
    #edit1.textChanged.connect(lambda:Print('Input value is {}'.format(self.edit1.text())))
    edit1.textChanged.connect(lambda txt:Print('Input value is {}'.format(txt)))
    edit1.move(10, 60)
    self.edit1= edit1

    # Show window
    self.show()

# Create an PyQT4 application object.
a = QtGui.QApplication(sys.argv)

# The QWidget widget is the base class of all user interface objects in PyQt4.
w = TLineEdit()

sys.exit(a.exec_())

