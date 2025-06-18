#!/usr/bin/python3
#\file    button06.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jun.18, 2025

import sys
import random
#from PyQt4 import QtCore, QtGui
from _import_qt import *

def Print(s):
  print(s)

class TButton(QtGui.QWidget):
  def __init__(self):
    QtGui.QWidget.__init__(self)
    self.InitUI()

  def InitUI(self):
    # Set window size.
    self.resize(320, 240)

    # Set window title
    self.setWindowTitle("Checkable button")

    # Add a button
    btn1= QtGui.QPushButton('Enable', self)
    #btn1.setFlat(True)
    btn1.setCheckable(True)
    btn1.setFocusPolicy(QtCore.Qt.NoFocus)
    btn1.setToolTip('Click to activate Exit button')
    btn1.clicked.connect(lambda b,btn1=btn1: Print('Enabled') if btn1.isChecked() else Print('Disabled'))
    btn1.toggled.connect(lambda checked,btn1=btn1: btn1.setText('Disable') if checked else btn1.setText('Enable'))
    #btn1.resize(btn1.sizeHint())
    btn1.move(70, 60)

    btn1.setSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
    #btn1.setMinimumWidth(btn1.minimumSizeHint().width()*0.5)
    #btn1.resize(btn1.minimumSizeHint().width(), btn1.sizeHint().height())
    #btn1.setContentsMargins(0, 0, 0, 0)
    btn1.setStyleSheet('padding:5px 10px 5px 10px; background-color: lightblue;')

    btn2= QtGui.QPushButton('Exit', self)
    btn2.setToolTip('Click to exit')
    btn2.resize(btn2.sizeHint())
    btn2.clicked.connect(lambda b,btn1=btn1: self.close() if btn1.isChecked() else Print('Activation button is not checked'))
    btn2.move(70, 150)
    btn2.setStyleSheet('padding:5px 10px 5px 10px; background-color: pink;')

    btn3= QtGui.QPushButton('ChangeColor', self)
    btn3.setToolTip('Click to change button color')
    btn3.resize(btn3.sizeHint())
    btn3.clicked.connect(lambda b, btn1=btn1, btn2=btn2, btn3=btn3: self.change_button_colors(btn1, btn2, btn3))
    btn3.move(180, 100)
    btn3.setStyleSheet('padding:5px 10px 5px 10px; background-color: yellow;')

    # Show window
    self.show()

  def change_button_colors(self, *buttons):
    COLORS= ['blue', 'red', 'green', 'yellow', 'pink', 'purple', 'orange', 'gray', 'brown', 'cyan']
    COLORS+= ['lightblue', 'lightgreen', 'lightyellow', 'lightpink', 'lavender', 'beige', 'mintcream', 'aliceblue', 'peachpuff', 'lightgray', 'honeydew', 'ivory', 'azure', 'powderblue', 'palegoldenrod']
    for btn in buttons:
      color= random.choice(COLORS)
      btn.setStyleSheet(f'padding:5px 10px 5px 10px; background-color: {color};')
      Print(f'{btn.text()} color changed to: {color}')

# Create an PyQT4 application object.
a = QtGui.QApplication(sys.argv)

# The QWidget widget is the base class of all user interface objects in PyQt4.
w = TButton()

sys.exit(a.exec_())
