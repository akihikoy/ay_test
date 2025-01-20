#!/usr/bin/python3
#\file    radiobutton4.py
#\brief   RadioBox class.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Apr.14, 2021

import sys
from PyQt4 import QtCore,QtGui

class TRadioBox(QtGui.QWidget):
  def __init__(self, *args, **kwargs):
    super(TRadioBox, self).__init__(*args, **kwargs)

  def Construct(self, layout, options, index, onclick, font_size):
    self.layout= None
    if layout=='h':  self.layout= QtGui.QHBoxLayout()
    elif layout=='v':  self.layout= QtGui.QVBoxLayout()
    elif layout=='grid':  raise Exception('Not implemented yet')
    else:  raise Exception('Invalid layout option:',layout)

    self.group= QtGui.QButtonGroup()
    self.radbtns= []
    for idx,option in enumerate(options):
      radbtn= QtGui.QRadioButton(option, self)
      radbtn.setCheckable(True)
      if idx==index:  radbtn.setChecked(True)
      radbtn.setFocusPolicy(QtCore.Qt.NoFocus)
      radbtn.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
      #radbtn.move(10, 60)
      if font_size:  radbtn.setFont(QtGui.QFont('', font_size))
      if onclick:  radbtn.clicked.connect(onclick)
      self.layout.addWidget(radbtn)
      self.group.addButton(radbtn)
      self.radbtns.append(radbtn)
    self.setLayout(self.layout)

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
    mainlayout= QtGui.QVBoxLayout()
    #mainlayout= QtGui.QGridLayout()
    self.setLayout(mainlayout)

    # Add a button
    btn1= QtGui.QPushButton('__Exit?__', self)
    #btn1.setFlat(True)
    btn1.setToolTip('Click to make something happen')
    btn1.clicked.connect(lambda:self.close() if self.radiobox.radbtns[1].isChecked() else Print('Hint: Select Option-2 to exit'))
    #btn1.resize(btn1.sizeHint())
    btn1.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
    btn1.move(100, 150)
    btn1.font_size= (10,30)
    btn1.setFont(QtGui.QFont('', btn1.font_size[0]))
    self.btn1= btn1
    mainlayout.addWidget(btn1)#,0,0)

    radiobox= TRadioBox(self)
    radiobox.font_size= (10,30)
    clicked= lambda: self.btn1.setText('Exit') if self.radiobox.radbtns[1].isChecked() else self.btn1.setText('Not exit({text})'.format(text=self.radiobox.group.checkedButton().text()))
    radiobox.Construct('h',['Option-1','Option-2','Option-3'], index=2, onclick=clicked, font_size=radiobox.font_size[0])
    self.radiobox= radiobox
    mainlayout.addWidget(radiobox)

    self.resizeEvent= lambda event: (self.ResizeText(self.btn1,event), self.ResizeText(self.radiobox,event),None)[-1]

    # Show window
    self.show()

  def ResizeText(self, obj, event):
    if isinstance(obj,TRadioBox):
      for radbtn in obj.radbtns:
        radbtn.font_size= obj.font_size
        self.ResizeText(radbtn, event)
      return
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
