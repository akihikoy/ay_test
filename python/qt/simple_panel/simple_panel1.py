#!/usr/bin/python
#\file    simple_panel1.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Apr.14, 2021

import sys
from PyQt4 import QtCore,QtGui

def InsertDict(d_base, d_new):
  for k_new,v_new in d_new.iteritems():
    if k_new in d_base and (type(v_new)==dict and type(d_base[k_new])==dict):
      InsertDict(d_base[k_new], v_new)
    else:
      d_base[k_new]= v_new

class TSimplePanel(QtGui.QWidget):
  def __init__(self, title, widgets, layout, size=(800,400)):
    QtGui.QWidget.__init__(self)
    self.InitUI(title, widgets, layout, size)

  #def ResizeText(self, obj, event):
    #font_size= min(obj.font_size_range[1],max(obj.font_size_range[0],int(self.rect().height()/100.*obj.font_size_range[0])))
    #f= QtGui.QFont('', font_size)
    #if isinstance(obj,QtGui.QRadioButton):
      #obj.setStyleSheet('QRadioButton::indicator {{width:{0}px;height:{0}px;}};'.format(1.3*font_size))
    #elif isinstance(obj,QtGui.QComboBox):
      #obj.resize(obj.sizeHint().width(),obj.height())
    #obj.setFont(f)

  def ResizeText(self, event):
    for name,obj in self.widgets.iteritems():
      font_size= min(obj.font_size_range[1],max(obj.font_size_range[0],int(self.rect().height()/100.*obj.font_size_range[0])))
      f= QtGui.QFont('', font_size)
      if isinstance(obj,QtGui.QRadioButton):
        obj.setStyleSheet('QRadioButton::indicator {{width:{0}px;height:{0}px;}};'.format(1.3*font_size))
      elif isinstance(obj,QtGui.QComboBox):
        obj.resize(obj.sizeHint().width(),obj.height())
      obj.setFont(f)

  def AddButton(self, w_param):
    param={
      'text': 'button',
      'onclick': None,
      'font_size_range': (10,30),
      }
    InsertDict(param, w_param)
    btn= QtGui.QPushButton(param['text'], self)
    btn.setFocusPolicy(QtCore.Qt.NoFocus)
    #btn.setFlat(True)
    #btn.setToolTip('Click to make something happen')
    if param['onclick']:  btn.clicked.connect(lambda btn=btn: param['onclick'](self,btn))
    btn.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
    btn.resize(btn.sizeHint())
    #btn.move(100, 150)
    btn.font_size_range= param['font_size_range']
    btn.setFont(QtGui.QFont('', btn.font_size_range[0]))
    #btn.resizeEvent= lambda event,obj=btn: self.ResizeText(obj,event)
    return btn

  def AddButtonCheckable(self, w_param):
    param={
      'text': ('button','button'),
      'checked': False,
      'onclick': None,
      'font_size_range': (10,30),
      }
    InsertDict(param, w_param)
    btn= QtGui.QPushButton(param['text'][0], self)
    btn.setFocusPolicy(QtCore.Qt.NoFocus)
    btn.setCheckable(True)
    btn.setChecked(param['checked'])
    if param['onclick']:  btn.clicked.connect(lambda bnt=btn: (param['onclick'][0](self,btn) if param['onclick'][0] else None, btn.setText(param['text'][1])) if btn.isChecked() else (param['onclick'][1](self,btn) if param['onclick'][1] else None, btn.setText(param['text'][0])) )
    btn.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
    btn.resize(btn.sizeHint())
    #btn.move(220, 100)
    btn.font_size_range= param['font_size_range']
    btn.setFont(QtGui.QFont('', btn.font_size_range[0]))
    #btn.resizeEvent= lambda event,obj=btn: self.ResizeText(obj,event)
    return btn

  def AddComboBox(self, w_param):
    param={
      'options': [],
      'index': 0,
      'onactivated': None,
      'font_size_range': (10,30),
      }
    InsertDict(param, w_param)
    cmbbx= QtGui.QComboBox(self)
    cmbbx.setFocusPolicy(QtCore.Qt.NoFocus)
    for option in param['options']:
      cmbbx.addItem(option)
    cmbbx.setCurrentIndex(param['index'])
    if param['onactivated']:  cmbbx.activated[str].connect(lambda _,cmbbx=cmbbx:param['onactivated'](self,cmbbx))
    cmbbx.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
    cmbbx.resize(cmbbx.sizeHint())
    #cmbbx.move(10, 60)
    cmbbx.font_size_range= param['font_size_range']
    cmbbx.setFont(QtGui.QFont('', cmbbx.font_size_range[0]))
    #cmbbx.resizeEvent= lambda event,obj=cmbbx: self.ResizeText(obj,event)
    return cmbbx

  def AddLineEdit(self, w_param):
    param={
      'validator': None,  #'int'
      'enabled': True,
      'font_size_range': (10,30),
      }
    InsertDict(param, w_param)
    edit= QtGui.QLineEdit(self)
    if param['validator']=='int':  edit.setValidator(QtGui.QIntValidator())
    edit.setMinimumHeight(10)
    edit.setMinimumWidth(10)
    edit.setEnabled(param['enabled'])
    edit.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Fixed)
    edit.resize(edit.sizeHint())
    #edit.move(10, 60)
    edit.font_size_range= param['font_size_range']
    edit.setFont(QtGui.QFont('', edit.font_size_range[0]))
    #edit.resizeEvent= lambda event,obj=edit: self.ResizeText(obj,event)
    return edit

  def AddRadioBoxH(self, w_param):
    param={
      'onclick': None,
      'font_size_range': (10,30),
      }
    InsertDict(param, w_param)
  def AddRadioBoxV(self, w_param):
    param={
      'onclick': None,
      'font_size_range': (10,30),
      }
    InsertDict(param, w_param)
  def AddSliderH(self, w_param):
    param={
      'range': (0,10,1),
      'font_size_range': (10,30),
      }
    InsertDict(param, w_param)
  def AddSpacer(self, w_param):
    param={
      }
    InsertDict(param, w_param)

  def AddLayouts(self, layout):
    l_type,name,items= layout
    if name is None:
      for i in range(10000):
        name= l_type+str(i)
        if name not in self.layouts:  break
    self.layouts[name]= None

    layout= None
    if l_type in ('boxv','boxh'):
      layout= QtGui.QVBoxLayout() if l_type=='boxv' else QtGui.QHBoxLayout()
      for item in items:
        if isinstance(item,str):
          widget= self.widgets[item]
          if isinstance(widget,QtGui.QSpacerItem):
            layout.addSpacerItem(widget)
          else:
            layout.addWidget(widget)
        else:
          sublayout= self.AddLayouts(item)
          layout.addLayout(sublayout)
    elif l_type=='grid':
      pass
    elif l_type=='tab':
      pass

    self.layouts[name]= layout
    return layout


  def InitUI(self, title, widgets, layout, size):
    self.resize(*size)  #window size
    self.setWindowTitle(title)

    self.widget_generator= {
      'button': self.AddButton,
      'buttonch': self.AddButtonCheckable,
      'combobox': self.AddComboBox,
      'lineedit': self.AddLineEdit,
      'radioboxh': self.AddRadioBoxH,
      'radioboxv': self.AddRadioBoxV,
      'sliderh': self.AddSliderH,
      'spacer': self.AddSpacer,
      }
    self.widgets_in= widgets
    self.widgets= {}
    for name,(w_type, w_param) in self.widgets_in.iteritems():
      self.widgets[name]= self.widget_generator[w_type](w_param)

    #self.layout_generator= {
      #'boxv': self.AddBoxV,
      #'boxh': self.AddBoxH,
      #'grid': self.AddGrid,
      #'tab': self.AddGrid,
      #}
    self.layout_in= layout
    self.layouts= {}
    self.setLayout(self.AddLayouts(self.layout_in))

    self.ResizeText(None)
    self.resizeEvent= self.ResizeText

    self.show()


if __name__=='__main__':
  def Print(*s):
    for ss in s:  print ss,
    print ''

  widgets= {
    'btn1': ('button',
             {'text':'Close',
              'onclick':lambda w,obj:w.close()}),
    'btn2': ('buttonch',
             {'text':('TurnOn','TurnOff'),
              'onclick': (lambda w,obj:Print('ON!'),
                          lambda w,obj:Print('OFF!'))}),
    'cmb1': ('combobox',
             {'options':('Option-0','Option-1','Option-2','Other'),
              'index':1,
              'onactivated': lambda w,obj:(Print('Selected',obj.currentText()),
                                           w.widgets['edit_cmb1other'].setEnabled(obj.currentText()=='Other'))}),
    'edit_cmb1other': ('lineedit',
                       {'validator':'int',
                        'enabled':False}),
    #'radbtn1': ('radioboxh', {'onclick': onclick_r1}),
    #'edit_radbtn1other': ('lineedit', {'validator': 'int'}),
    #'radbtn2': ('radioboxv', {'onclick': onclick_r2}),
    #'edit_radbtn2other': ('lineedit', {'validator': 'int'}),
    #'slider1': ('sliderh', {'range': (1000,1800,100)}),
    #'spacer1': ('spacer', {}),
    }
  #Layout option: vbox, hbox, grid, tab
  #layout= ('boxv',None,
           #(
             #('grid',None, (('btn1',0,0),('btn2',0,1,1,2),
                            #('spacer1',1,0),('cmb1',1,1),('edit_cmb1other',1,2)) ),
             #('boxh',None, ('radbtn1','edit_radbtn1other')),
             #('boxv',None, ('radbtn2','edit_radbtn2other')),
             #'slider1',
           #))
  layout= ('boxv',None,
           (
             ('boxh',None, ('btn1','btn2','cmb1','edit_cmb1other') ),
           ))

  app= QtGui.QApplication(sys.argv)
  win= TSimplePanel('Simple Panel', widgets, layout)
  sys.exit(app.exec_())
