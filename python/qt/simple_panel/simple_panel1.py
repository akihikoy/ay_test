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

class TRadioBox(QtGui.QWidget):
  def __init__(self, *args, **kwargs):
    super(TRadioBox, self).__init__(*args, **kwargs)

  #layout: 'h'(horizontal), 'v'(vertical), 'grid'
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

  def setFont(self, f):
    for radbtn in self.radbtns:
      radbtn.setFont(f)


class TSimplePanel(QtGui.QWidget):
  def __init__(self, title, widgets, layout, size=(800,400), font_height_scale=100.0):
    QtGui.QWidget.__init__(self)
    self.font_height_scale= font_height_scale
    self.InitUI(title, widgets, layout, size)

  def ResizeTextOfObj(self, obj, font_size_range, size):
    font_size= min(font_size_range[1],max(font_size_range[0],int(size*font_size_range[0])))
    f= QtGui.QFont('', font_size)
    if isinstance(obj,(QtGui.QRadioButton,TRadioBox)):
      obj.setStyleSheet('QRadioButton::indicator {{width:{0}px;height:{0}px;}};'.format(1.3*font_size))
      #obj.resize(obj.sizeHint().width(),obj.height())
    elif isinstance(obj,QtGui.QLineEdit):
      #obj.resize(obj.sizeHint().width(),obj.height())
      text= obj.text()
      rect= obj.fontMetrics().boundingRect(text if text!='' else '0')
      obj.setMinimumWidth(rect.width()+15)
      obj.setMinimumHeight(rect.height()+15)
    elif isinstance(obj,QtGui.QComboBox):
      obj.resize(obj.sizeHint().width(),obj.height())
    obj.setFont(f)

  def ResizeText(self, event):
    s= self.rect().height()/self.font_height_scale
    for name,obj in self.widgets.iteritems():
      self.ResizeTextOfObj(obj, obj.font_size_range, s)

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
    if param['index'] is not None:  cmbbx.setCurrentIndex(param['index'])
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

  def AddRadioBox(self, w_param):
    param={
      'options': [],
      'index': None,
      'layout': 'h',  #h(horizontal),v(vertical),grid
      'onclick': None,
      'font_size_range': (10,30),
      }
    InsertDict(param, w_param)

    radiobox= TRadioBox(self)
    radiobox.font_size_range= param['font_size_range']
    if param['onclick']:  clicked= lambda _,radiobox=radiobox:param['onclick'](self,radiobox)
    else:  clicked= None
    radiobox.Construct(param['layout'], param['options'], index=param['index'], onclick=clicked, font_size=radiobox.font_size_range[0])

    return radiobox

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
      'radiobox': self.AddRadioBox,
      'sliderh': self.AddSliderH,
      'spacer': self.AddSpacer,
      }
    self.widgets_in= widgets
    self.widgets= {}
    for name,(w_type, w_param) in self.widgets_in.iteritems():
      self.widgets[name]= self.widget_generator[w_type](w_param)

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
    'radbox1': ('radiobox',
                {'options':('Option-0','Option-1','Option-2','Other'),
                 'layout': 'h',
                 'index': 0,
                 'onclick': lambda w,obj:(Print('Selected',obj.group.checkedButton().text()),
                                           w.widgets['edit_radbox1other'].setEnabled(obj.group.checkedButton().text()=='Other'))}),
    'edit_radbox1other': ('lineedit',
                          {'validator':'int',
                           'enabled':False}),
    'radbox2': ('radiobox',
                {'options':('Option-A','Option-B','Option-C','Other'),
                 'layout': 'v',
                 'index': None,
                 'onclick': lambda w,obj:(Print('Selected',obj.group.checkedButton().text()),
                                           w.widgets['edit_radbox2other'].setEnabled(obj.group.checkedButton().text()=='Other'))}),
    'edit_radbox2other': ('lineedit',
                          {'validator':'int',
                           'enabled':False}),
    #'slider1': ('sliderh', {'range': (1000,1800,100)}),
    #'spacer1': ('spacer', {}),
    }
  #Layout option: vbox, hbox, grid, tab
  #layout= ('boxv',None,
           #(
             #('grid',None, (('btn1',0,0),('btn2',0,1,1,2),
                            #('spacer1',1,0),('cmb1',1,1),('edit_cmb1other',1,2)) ),
             #('boxh',None, ('radbox1','edit_radbox1other')),
             #('boxv',None, ('radbox2','edit_radbox2other')),
             #'slider1',
           #))
  layout= ('boxv',None,
           (
             ('boxh',None, ('btn1','btn2','cmb1','edit_cmb1other') ),
             ('boxh',None, ('radbox1','edit_radbox1other') ),
             ('boxv',None, ('radbox2','edit_radbox2other') ),
           ))

  app= QtGui.QApplication(sys.argv)
  win= TSimplePanel('Simple Panel', widgets, layout, size=(600,300), font_height_scale=200.0)
  sys.exit(app.exec_())
