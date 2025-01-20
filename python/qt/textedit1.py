#!/usr/bin/python3
#\file    textedit1.py
#\brief   Test QTextEdit
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Apr.14, 2021

import sys
from PyQt4 import QtCore,QtGui

def Print(*s):
  for ss in s:  print(ss, end=' ')
  print('')

class TTextEdit(QtGui.QWidget):
  def __init__(self):
    QtGui.QWidget.__init__(self)
    self.InitUI()

  def InitUI(self):
    # Set window size.
    self.resize(320, 240)

    # Set window title
    self.setWindowTitle("TextEdit")

    # Add a button
    btn1= QtGui.QPushButton('_________Exit?_________', self)
    #btn1.setFlat(True)
    btn1.setToolTip('Click to make something happen')
    btn1.clicked.connect(lambda:self.close() if 1+self.text1.toPlainText().count('\n')>2 else Print('Hint: Write more than 2 lines to exit; currently:',1+self.text1.toPlainText().count('\n')))
    btn1.resize(btn1.sizeHint())
    btn1.move(100, 150)
    self.btn1= btn1

    text1= QtGui.QTextEdit(self)
    text1.setReadOnly(True)
    text1.move(10, 60)
    text1.resize(200, 80)
    self.text1= text1

    btn2= QtGui.QPushButton('To editable', self)
    btn2.setCheckable(True)
    btn2.setChecked(False)
    btn2.setFocusPolicy(QtCore.Qt.NoFocus)
    btn2.clicked.connect(lambda: (Print('Enabled'),self.text1.setReadOnly(False),btn2.setText('To readonly')) if btn2.isChecked() else (Print('Disabled'),self.text1.setReadOnly(True),btn2.setText('To editable')))
    btn2.resize(btn2.sizeHint())
    btn2.move(220, 100)

    # Show window
    self.show()

# Create an PyQT4 application object.
a = QtGui.QApplication(sys.argv)

# The QWidget widget is the base class of all user interface objects in PyQt4.
w = TTextEdit()

sys.exit(a.exec_())

