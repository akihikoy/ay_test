#!/usr/bin/python3
#\file    radiobutton2.py
#\brief   Test expanding radio button.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Apr.13, 2021

import sys
from PyQt4 import QtCore,QtGui

def Print(s):
  print(s)

class TRadioButton(QtGui.QWidget):
  def __init__(self):
    QtGui.QWidget.__init__(self)
    self.InitUI()

  def InitUI(self):
    # Set window size.
    self.resize(320, 240)

    # Set window title
    self.setWindowTitle("Radio button")

    # Horizontal box layout
    mainlayout= QtGui.QHBoxLayout()
    #mainlayout= QtGui.QGridLayout()
    self.setLayout(mainlayout)

    # Add a button
    btn1= QtGui.QPushButton('__Exit?__', self)
    #btn1.setFlat(True)
    btn1.setToolTip('Click to make something happen')
    btn1.clicked.connect(lambda:self.close() if self.radbtn2.isChecked() else Print('Hint: Select Option-2 to exit'))
    #btn1.resize(btn1.sizeHint())
    btn1.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
    btn1.move(100, 150)
    btn1.font_size= (10,30)
    btn1.setFont(QtGui.QFont('', btn1.font_size[0]))
    btn1.resizeEvent= lambda event,obj=btn1: self.ResizeText(obj,event)
    self.btn1= btn1
    mainlayout.addWidget(btn1)#,0,0)

    radbtn1= QtGui.QRadioButton('Option-1', self)
    radbtn1.setCheckable(True)
    radbtn1.setFocusPolicy(QtCore.Qt.NoFocus)
    radbtn1.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
    radbtn1.move(10, 60)
    radbtn1.font_size= (10,30)
    radbtn1.setFont(QtGui.QFont('', radbtn1.font_size[0]))
    radbtn1.resizeEvent= lambda event,obj=radbtn1: self.ResizeText(obj,event)
    radbtn1.setChecked(True)
    mainlayout.addWidget(radbtn1)#,0,1)

    radbtn2= QtGui.QRadioButton('Option-2', self)
    radbtn2.setCheckable(True)
    radbtn2.setFocusPolicy(QtCore.Qt.NoFocus)
    radbtn2.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
    self.radbtn2= radbtn2
    radbtn2.move(100, 60)
    radbtn2.font_size= (10,30)
    radbtn2.setFont(QtGui.QFont('', radbtn2.font_size[0]))
    radbtn2.resizeEvent= lambda event,obj=radbtn2: self.ResizeText(obj,event)
    mainlayout.addWidget(radbtn2)#,0,2)

    radbtn3= QtGui.QRadioButton('Option-3', self)
    radbtn3.setCheckable(True)
    radbtn3.setFocusPolicy(QtCore.Qt.NoFocus)
    radbtn3.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
    radbtn3.move(190, 60)
    radbtn3.font_size= (10,30)
    radbtn3.setFont(QtGui.QFont('', radbtn3.font_size[0]))
    radbtn3.resizeEvent= lambda event,obj=radbtn3: self.ResizeText(obj,event)
    mainlayout.addWidget(radbtn3)#,0,3)

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

  def ResizeText(self, obj, event):
    font_size= min(obj.font_size[1],max(obj.font_size[0],int(self.rect().height()/100.*obj.font_size[0])))
    f= QtGui.QFont('', font_size)
    if isinstance(obj,QtGui.QRadioButton):
      obj.setStyleSheet('QRadioButton::indicator {{width:{0}px;height:{0}px;}};'.format(1.3*font_size))
    obj.setFont(f)

# Create an PyQT4 application object.
a = QtGui.QApplication(sys.argv)

# The QWidget widget is the base class of all user interface objects in PyQt4.
w = TRadioButton()

sys.exit(a.exec_())
