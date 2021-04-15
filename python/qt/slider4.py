#!/usr/bin/python
#\file    slider4.py
#\brief   New QWidget slider class
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Apr.15, 2021

import sys
from PyQt4 import QtCore,QtGui

class TSlider(QtGui.QWidget):
  def __init__(self, *args, **kwargs):
    super(TSlider, self).__init__(*args, **kwargs)

  def convert_from(self, slider_value):
    return min(self.range_step[1], self.range_step[0] + self.range_step[2]*slider_value)

  def convert_to(self, value):
    return max(0,min(self.slider_max,(value-self.range_step[0])/self.range_step[2]))

  def value(self):
    return self.convert_from(self.slider.value())

  def setValue(self, value):
    slider_value= self.convert_to(value)
    self.slider.setValue(slider_value)
    self.setLabel(value)

  def setLabel(self, value):
    self.label.setText(str(value).rjust(len(str(self.range_step[1]))))

  #style: 0:Default, 1:Variable handle size.
  def Construct(self, range_step, n_labels, slider_style, onvaluechange):
    self.range_step= range_step
    self.slider_max= (self.range_step[1]-self.range_step[0])/self.range_step[2]
    self.slider_style= slider_style

    self.layout= QtGui.QGridLayout()

    vspacer1= QtGui.QSpacerItem(1, 1, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
    self.layout.addItem(vspacer1, 0, 0, 1, n_labels+1)

    self.slider= QtGui.QSlider(QtCore.Qt.Horizontal, self)
    #self.slider.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
    self.slider.setTickPosition(QtGui.QSlider.TicksBothSides)
    self.slider.setRange(0, self.slider_max)
    self.slider.setTickInterval(1)
    self.slider.setSingleStep(1)
    #self.slider.move(10, 60)
    #self.slider.resize(100, 20)
    self.slider.valueChanged.connect(lambda *args,**kwargs:(self.setLabel(self.value()), onvaluechange(*args,**kwargs) if onvaluechange else None)[-1])

    self.layout.addWidget(self.slider, 1, 0, 1, n_labels)

    self.label= QtGui.QLabel('0',self)
    self.layout.addWidget(self.label, 1, n_labels, 1, 1, QtCore.Qt.AlignLeft)

    #hspacer1= QtGui.QSpacerItem(1, 1, QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.MinimumExpanding)
    #self.layout.addItem(hspacer1, 1, n_labels+1)

    self.tick_labels= []
    if n_labels>1:
      #tick_font= QtGui.QFont(self.label.font().family(), self.label.font().pointSize()*0.6)
      label_step= (range_step[1]-range_step[0])/(n_labels-1)
      for i_label in range(n_labels):
        label= str(range_step[0]+i_label*label_step)
        tick_label= QtGui.QLabel(label,self)
        #tick_label.setFont(tick_font)
        if i_label<(n_labels-1)/2:  align= QtCore.Qt.AlignLeft
        elif i_label==(n_labels-1)/2:  align= QtCore.Qt.AlignCenter
        else:  align= QtCore.Qt.AlignRight
        self.layout.addWidget(tick_label, 2, i_label, 1, 1, align)
        self.tick_labels.append(tick_label)

    vspacer2= QtGui.QSpacerItem(1, 1, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
    self.layout.addItem(vspacer2, 3, 0, 1, n_labels+1)
    self.setValue(range_step[0])
    self.setLayout(self.layout)
    self.setStyleForFont(self.label.font())

  def setStyleForFont(self, f):
    tick_f= QtGui.QFont(f.family(), f.pointSize()*0.6)
    for tick_label in self.tick_labels:
      tick_label.setFont(tick_f)
    if self.slider_style==0:
      self.slider.setStyleSheet('')
    elif self.slider_style==1:
      h0= f.pointSize()*2
      h1= h0+8
      self.slider.setStyleSheet('''
        QSlider {{
            height: {1}px;
        }}
        QSlider::groove:horizontal {{
            background: transparent;
            border: 2px solid #aaa;
            height: {0}px;
            margin: 0 0;
        }}
        QSlider::handle:horizontal {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #b4b4b4, stop:1 #8f8f8f);
            border: 1px solid #5c5c5c;
            width: {0}px;
            margin: 0 0;
            border-radius: 3px;
        }}
        '''.format(h0,h1))

  def setFont(self, f):
    self.label.setFont(f)
    self.setStyleForFont(f)


def Print(*s):
  for ss in s:  print ss,
  print ''

class TSliderTest(QtGui.QWidget):
  def __init__(self):
    QtGui.QWidget.__init__(self)
    self.InitUI()

  def InitUI(self):
    # Set window size.
    self.resize(320, 120)

    # Set window title
    self.setWindowTitle("SliderTest")

    mainlayout= QtGui.QVBoxLayout()
    self.setLayout(mainlayout)

    slider1= TSlider(self)
    slider1.Construct([1000,1800,100], n_labels=5, slider_style=1, onvaluechange=lambda _:Print(slider1.value()))
    slider1.setValue(1600)
    slider1.font_size= (10,30)
    slider1.setFont(QtGui.QFont('', slider1.font_size[0]))
    self.slider1= slider1
    mainlayout.addWidget(slider1)

    # Add a button
    btn1= QtGui.QPushButton('_________Exit?_________', self)
    #btn1.setFlat(True)
    btn1.setToolTip('Click to make something happen')
    btn1.clicked.connect(lambda:self.close() if self.slider1.value()<1500 else Print('Hint: Set value less than 1500 to exit'))
    btn1.resize(btn1.sizeHint())
    #btn1.move(100, 150)
    btn1.font_size= (10,30)
    btn1.setFont(QtGui.QFont('', btn1.font_size[0]))
    #btn1.resizeEvent= lambda event,obj=btn1: self.ResizeText(obj,event)
    self.btn1= btn1
    mainlayout.addWidget(btn1)

    btn1.resizeEvent= lambda event,objs=(btn1,slider1): ([self.ResizeText(obj,event) for obj in objs]+[None])[-1]

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
w = TSliderTest()

sys.exit(a.exec_())
