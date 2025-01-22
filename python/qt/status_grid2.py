#!/usr/bin/python3
#\file    status_grid2.py
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
  def __init__(self, *args, **kwargs):
    super(TStatusGrid, self).__init__(*args, **kwargs)
    #self.setMinimumSize(100, 100)
    self.setBackgroundRole(QtGui.QPalette.Base)

  '''Construct the status grid widget.
    list_status: List of status items (dictionaries) (this class also modified the content).
    direction: Direction to list in the grid ('vertical' or 'horizontal').
    shape: Shape of 'color' item ('circle', 'square').
    rows,columns: Number of rows and columns. At least one of it should be specified.
  '''
  def Construct(self, list_status, direction='vertical', shape='circle', rows=None, columns=3):
    # Grid layout
    self.grid= QtGui.QGridLayout()

    # Define a list of status items.
    self.list_status= list_status
    self.dict_status= {item['label']: item for item in self.list_status}

    if rows is None and columns is None:  raise Exception('TStatusGrid: one of rows or columns should be None.')
    if rows is not None and columns is not None: assert(len(self.list_status)<rows*columns)
    if rows is None:  rows= int(math.ceil(float(len(self.list_status))/columns))
    if columns is None:  columns= int(math.ceil(float(len(self.list_status))/rows))
    self.rows,self.columns= rows,columns

    self.direction= direction
    self.shape= shape

    #self.default_font_size= 10
    if self.direction=='vertical':
      for c in range(self.columns):
        for r in range(self.rows):
          idx= c*self.rows + r
          if idx>=len(self.list_status):  break
          self.AddWidgetsForItem(self.list_status[idx], r, c)
    elif self.direction=='horizontal':
      for r in range(self.rows):
        for c in range(self.columns):
          idx= r*self.columns + c
          if idx>=len(self.list_status):  break
          self.AddWidgetsForItem(self.list_status[idx], r, c)
    else:
      raise Exception('TStatusGrid: invalid direction:',self.direction)

    self.setLayout(self.grid)

  def AddWidgetsForItem(self, item, r, c):
    if item['type']=='color':
      color1= TRenderPrimitive(self.shape, (0.05,0.05), self.StateColor(item), self)
      color1.setSizePolicy(QtGui.QSizePolicy.Maximum, QtGui.QSizePolicy.Expanding)
      color1.min_size= 20
      color1.draw_bevel= False
      item['w_color']= color1

      label1= QtGui.QLabel(item['label'], self)
      label1.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
      label1.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
      #label1.font_size= (8,28)
      #label1.setFont(QtGui.QFont('', self.default_font_size))
      #label1.resizeEvent= lambda event,obj=label1: self.ResizeText(obj,event)
      item['w_label']= label1

      rowsize,colsize= 1,1
      self.grid.addWidget(color1, r, 2*c, rowsize, colsize)
      self.grid.addWidget(label1, r, 2*c+1, rowsize, colsize)

    elif item['type']=='text':
      label1= QtGui.QLabel(self.StateText(item), self)
      label1.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
      label1.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
      #label1.font_size= (8,28)
      #label1.setFont(QtGui.QFont('', self.default_font_size))
      #label1.resizeEvent= lambda event,obj=label1: self.ResizeText(obj,event)
      item['w_label']= label1

      rowsize,colsize= 1,2
      self.grid.addWidget(label1, r, 2*c, rowsize, colsize)

  def StateColor(self, item):
    state= item['state']
    col_map= {'green': (0,255,0), 'yellow': (255,255,0), 'red': (255,0,0)}
    if state in col_map:  return col_map[state]
    return (128,128,128)

  def StateText(self, item):
    state= item['state']
    return ' {}: {}'.format(item['label'],item['state'])

  def UpdateStatus(self, label, state):
    item= self.dict_status[label]
    item['state']= state
    if item['type']=='color':
      item['w_color'].color= self.StateColor(item)
      item['w_color'].update()
    elif item['type']=='text':
      item['w_label'].setText(self.StateText(item))

  def setFont(self, f):
    for item in self.list_status:
      if 'w_label' in item:  item['w_label'].setFont(f)


def Print(s):
  print(s)

class TStatusGridTest(QtGui.QWidget):
  def __init__(self):
    QtGui.QWidget.__init__(self)
    self.InitUI()

  def InitUI(self):
    # Set window size.
    self.resize(400, 240)
    self.font_height_scale= 200

    # Set window title
    self.setWindowTitle("StatusGridTest")

    mainlayout= QtGui.QHBoxLayout()
    self.setLayout(mainlayout)

    list_status= [
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

    statusgrid1= TStatusGrid(self)
    #statusgrid1.move(10, 10)
    #statusgrid1.resize(240, 240)
    statusgrid1.Construct(list_status, columns=3, direction='vertical', shape='square')
    statusgrid1.font_size_range= (8,28)
    self.statusgrid1= statusgrid1
    mainlayout.addWidget(statusgrid1)

    sublayout= QtGui.QVBoxLayout()
    mainlayout.addLayout(sublayout)

    cmbbx1= QtGui.QComboBox(self)
    for item in list_status:
      cmbbx1.addItem(item['label'])
    cmbbx1.setCurrentIndex(1)
    #cmbbx1.move(260, 20)
    cmbbx1.font_size_range= (8,28)
    self.cmbbx1= cmbbx1
    sublayout.addWidget(cmbbx1)

    cmbbx2= QtGui.QComboBox(self)
    cmbbx2.addItem('green')
    cmbbx2.addItem('yellow')
    cmbbx2.addItem('red')
    cmbbx2.addItem('10')
    cmbbx2.addItem('100')
    cmbbx2.addItem('1000')
    cmbbx2.addItem('True')
    cmbbx2.addItem('False')
    cmbbx2.setCurrentIndex(1)
    #cmbbx2.move(260, 70)
    cmbbx2.font_size_range= (8,28)
    self.cmbbx2= cmbbx2
    sublayout.addWidget(cmbbx2)

    # Add a button
    btn1= QtGui.QPushButton('Update', self)
    #btn1.setFlat(True)
    btn1.setToolTip('Update state')
    btn1.clicked.connect(lambda:self.statusgrid1.UpdateStatus(str(self.cmbbx1.currentText()), str(self.cmbbx2.currentText())))
    #btn1.move(260, 120)
    btn1.font_size_range= (8,28)
    self.btn1= btn1
    sublayout.addWidget(btn1)

    # Add a button
    btn2= QtGui.QPushButton('Exit', self)
    #btn2.setFlat(True)
    btn2.clicked.connect(lambda:self.close())
    #btn2.move(260, 190)
    btn2.font_size_range= (8,28)
    self.btn2= btn2
    sublayout.addWidget(btn2)

    self.resizeEvent= self.ResizeText

    # Show window
    self.show()

  def ResizeText(self, event):
    size= self.rect().height()/float(self.font_height_scale)
    for obj in (self.statusgrid1,self.cmbbx1,self.cmbbx2,self.btn1,self.btn2):
      if not hasattr(obj,'font_size_range'):  continue
      font_size_range= obj.font_size_range
      font_size= min(font_size_range[1],max(font_size_range[0],int(size*font_size_range[0])))
      f= QtGui.QFont('', font_size)
      if hasattr(obj,'setFont'):
        obj.setFont(f)
        #print obj,f,font_size


if __name__=='__main__':
  # Create an PyQT4 application object.
  a = QtGui.QApplication(sys.argv)

  # The QWidget widget is the base class of all user interface objects in PyQt4.
  w = TStatusGridTest()

  sys.exit(a.exec_())

