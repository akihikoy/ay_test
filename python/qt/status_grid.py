#!/usr/bin/python3
#\file    status_grid.py
#\brief   Showing a list of status items with visual icons on a grid.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Apr.27, 2023

import sys
import math
#from PyQt4 import QtCore,QtGui
from _import_qt import *
from draw_primitive1 import TRenderPrimitive

def Print(s):
  print(s)

class TStatusGrid(QtGui.QWidget):
  def __init__(self):
    QtGui.QWidget.__init__(self)
    self.InitUI()

  def InitUI(self):
    # Set window size.
    self.resize(320, 240)

    # Set window title
    self.setWindowTitle("StatusGrid")

    # Grid layout
    grid= QtGui.QGridLayout()
    self.setLayout(grid)

    # Define a list of status items.
    self.list_status= [
      dict(label='Safety', type='color', state='yellow'),
      dict(label='URcomm', type='color', state='green'),
      dict(label='Robot', type='color', state='green'),
      dict(label='URRos', type='color', state='yellow'),
      dict(label='ScriptCore', type='color', state='yellow'),
      dict(label='FV',         type='color', state='green'),
      dict(label='Jetson',      type='color', state='green'),
      dict(label='Segmentation', type='color', state='green'),
      dict(label='GraspPlan',    type='color', state='green'),
      dict(label='PickMotion',    type='color', state='red'),
      dict(label='PlaceMotion',   type='color', state='red'),
      dict(label='LowItemDetect', type='color', state='red'),
      dict(label='ItemCount', type='text', state='10'),
      dict(label='Grasped', type='text', state='True'),
      ]

    columns= 3
    rows= int(math.ceil(float(len(self.list_status))/columns))
    ## Add buttons on grid
    #names= [[('a0\na01',lambda:Print('0')), ('a1',lambda:Print('1'))],
            #[('b0',lambda:Print('x')), ('b1',lambda:Print('y')), ('b2',lambda:Print('z'))],
            #[('c0',exit,2)]]
    #self.labels= []
    #self.buttons= []
    self.default_font_size= 10
    for c in range(columns):
      for r in range(rows):
        idx= c*rows + r
        if idx>=len(self.list_status):  break
        item= self.list_status[idx]

        #name,f,colsize= contents if len(contents)==3 else (contents[0],contents[1],1)
        #btn= QtGui.QPushButton(name)
        #btn.clicked.connect(f)
        #btn.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        #btn.setFont(QtGui.QFont('', self.default_font_size))
        #btn.resizeEvent= self.ResizeText

        if item['type']=='color':
          color1= TRenderPrimitive('circle', (0.05,0.05), self.StateColor(item['state']), self)
          color1.setSizePolicy(QtGui.QSizePolicy.Maximum, QtGui.QSizePolicy.Expanding)
          color1.min_size= 20
          color1.draw_bevel= False
          item['w_color']= color1

          label1= QtGui.QLabel(item['label'], self)
          label1.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
          label1.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
          label1.font_size= (8,28)
          label1.setFont(QtGui.QFont('', self.default_font_size))
          label1.resizeEvent= lambda event,obj=label1: self.ResizeText(obj,event)
          item['w_label']= label1

          rowsize,colsize= 1,1
          grid.addWidget(color1, r, 2*c, rowsize, colsize)
          grid.addWidget(label1, r, 2*c+1, rowsize, colsize)

        elif item['type']=='text':
          label1= QtGui.QLabel(' {}: {}'.format(item['label'],item['state']), self)
          label1.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
          label1.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
          label1.font_size= (8,28)
          label1.setFont(QtGui.QFont('', self.default_font_size))
          label1.resizeEvent= lambda event,obj=label1: self.ResizeText(obj,event)
          item['w_label']= label1

          rowsize,colsize= 1,2
          grid.addWidget(label1, r, 2*c, rowsize, colsize)

    # Show window
    self.show()

  def StateColor(self, state):
    col_map= {'green': (0,255,0), 'yellow': (255,255,0), 'red': (255,0,0)}
    if state in col_map:  return col_map[state]
    return (128,128,128)

  def ResizeText(self, obj, event):
    font_size= min(obj.font_size[1],max(obj.font_size[0],int(self.rect().height()/100.*obj.font_size[0])))
    f= QtGui.QFont('', font_size)
    obj.setFont(f)


if __name__=='__main__':
  # Create an PyQT4 application object.
  a = QtGui.QApplication(sys.argv)

  # The QWidget widget is the base class of all user interface objects in PyQt4.
  w = TStatusGrid()

  sys.exit(a.exec_())

