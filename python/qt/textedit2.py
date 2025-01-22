#!/usr/bin/python3
#\file    textedit2.py
#\brief   Expanding QTextEdit
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Apr.14, 2021

import sys
#from PyQt4 import QtCore,QtGui
from _import_qt import *

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

    mainlayout= QtGui.QVBoxLayout()
    #mainlayout= QtGui.QGridLayout()
    self.setLayout(mainlayout)

    text1= QtGui.QTextEdit(self)
    text1.setReadOnly(True)
    #text1.move(10, 60)
    text1.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Preferred)
    text1.setMinimumHeight(20)
    text1.setMinimumWidth(10)
    #text1.setFixedHeight(100)
    #text1.setMaximumHeight(1000)
    #text1.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
    #text1.resize(200, 100)
    #text1.resize(text1.sizeHint())
    text1.font_size= (10,30)
    text1.setFont(QtGui.QFont('', text1.font_size[0]))
    #NOTE: resizeEvent does not work.
    #text1.resizeEvent= lambda event,obj=text1: self.ResizeText(obj,event)
    #text1.resizeEvent= lambda event: text1.setFixedHeight(text1.document().size().toSize().height()+10)
    #NOTE: Uncomment the following lines to fit the text box size to the content.
    #text1.textChanged.connect(lambda: text1.setFixedHeight(text1.document().size().toSize().height()+10))
    #text1.textChanged.connect(lambda: text1.setMinimumHeight(text1.document().size().toSize().height()+10))
    self.text1= text1
    mainlayout.addWidget(text1)

    vspacer1= QtGui.QSpacerItem(20, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
    mainlayout.addSpacerItem(vspacer1)

    btn2= QtGui.QPushButton('To editable', self)
    btn2.setCheckable(True)
    btn2.setChecked(False)
    btn2.setFocusPolicy(QtCore.Qt.NoFocus)
    btn2.clicked.connect(lambda: (Print('Enabled'),self.text1.setReadOnly(False),btn2.setText('To readonly')) if btn2.isChecked() else (Print('Disabled'),self.text1.setReadOnly(True),btn2.setText('To editable')))
    btn2.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
    btn2.resize(btn2.sizeHint())
    #btn2.move(220, 100)
    btn2.font_size= (10,30)
    btn2.setFont(QtGui.QFont('', btn2.font_size[0]))
    btn2.resizeEvent= lambda event,obj=btn2: (self.ResizeText(obj,event), self.ResizeText(text1,event), None)[-1]
    mainlayout.addWidget(btn2)

    # Add a button
    btn1= QtGui.QPushButton('__Exit?__', self)
    #btn1.setFlat(True)
    btn1.setToolTip('Click to make something happen')
    btn1.clicked.connect(lambda:self.close() if 1+self.text1.toPlainText().count('\n')>2 else Print('Hint: Write more than 2 lines to exit; currently:',1+self.text1.toPlainText().count('\n')))
    btn1.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
    btn1.resize(btn1.sizeHint())
    btn1.move(100, 150)
    btn1.font_size= (10,30)
    btn1.setFont(QtGui.QFont('', btn1.font_size[0]))
    btn1.resizeEvent= lambda event,obj=btn1: self.ResizeText(obj,event)
    self.btn1= btn1
    mainlayout.addWidget(btn1)

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
w = TTextEdit()

sys.exit(a.exec_())

