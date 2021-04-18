#!/usr/bin/python
# -*- coding: utf-8 -*-
#\file    simple_panel1.py
#\brief   Qt-based simple panel designer.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Apr.14, 2021

import sys, copy
from PyQt4 import QtCore,QtGui

def MergeDict(d_base, d_new):
  for k_new,v_new in d_new.iteritems():
    if k_new in d_base and (type(v_new)==dict and type(d_base[k_new])==dict):
      MergeDict(d_base[k_new], v_new)
    else:
      d_base[k_new]= v_new
  return d_base  #NOTE: d_base is overwritten. Returning it is for the convenience.

def MergeDict2(d_base, *d_new):
  for d in d_new:
    MergeDict(d_base, d)
  return d_base  #NOTE: d_base is overwritten. Returning it is for the convenience.

class TRadioBox(QtGui.QWidget):
  def __init__(self, *args, **kwargs):
    super(TRadioBox, self).__init__(*args, **kwargs)

  #layout: 'h'(horizontal), 'v'(vertical), 'grid'
  def Construct(self, layout, options, index, onclick):
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
      #radbtn.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Fixed)
      radbtn.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
      #radbtn.move(10, 60)
      if onclick:  radbtn.clicked.connect(onclick)
      self.layout.addWidget(radbtn)
      self.group.addButton(radbtn)
      self.radbtns.append(radbtn)
    self.setLayout(self.layout)

  def setFont(self, f):
    for radbtn in self.radbtns:
      radbtn.setFont(f)
    h= f.pointSize()*1.5
    self.setStyleSheet('QRadioButton::indicator {{width:{0}px;height:{0}px;}};'.format(h))

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
    self.slider.setPageStep(1)
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

class TPrimitivePainter(QtGui.QWidget):
  def __init__(self, shape, margin, color, parent=None):
    super(TPrimitivePainter, self).__init__(parent)

    self.shape= shape
    self.margin= margin  #(horizontal_margin(ratio),vertical_margin(ratio))
    self.color= color
    self.setBackgroundRole(QtGui.QPalette.Base)
    #self.setAutoFillBackground(True)

    size_policy= QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
    size_policy.setHeightForWidth(True)
    self.setSizePolicy(size_policy)

  def setRandomColor(self):
    self.color= [255*random.random(),255*random.random(),255*random.random()]

  def setShape(self, shape):
    self.shape= shape

  def setMargin(self, margin):
    self.shape= margin

  def minimumSizeHint(self):
    return QtCore.QSize(100, self.heightForWidth(100))

  def sizeHint(self):
    return QtCore.QSize(400, self.heightForWidth(400))

  def heightForWidth(self, width):
    return width*1.2

  def paintEvent(self, event):
    col1= QtGui.QColor(*self.color)
    col2= QtGui.QColor(0.6*self.color[0], 0.6*self.color[1], 0.6*self.color[2])
    linear_gradient= QtGui.QLinearGradient(0, 0, self.width(), self.height())
    linear_gradient.setColorAt(0.0, QtCore.Qt.white)
    linear_gradient.setColorAt(0.2, col1)
    linear_gradient.setColorAt(0.8, col2)
    linear_gradient.setColorAt(1.0, QtCore.Qt.black)

    painter= QtGui.QPainter(self)
    painter.setPen(QtCore.Qt.SolidLine)
    painter.setBrush(linear_gradient)
    painter.setRenderHint(QtGui.QPainter.Antialiasing)

    painter.save()
    painter.translate(0, 0)

    if self.shape in ('ellipse','rect'):
      rect= QtCore.QRect(self.width()*self.margin[0], self.height()*self.margin[1], self.width()*(1.0-2.0*self.margin[0]), self.height()*(1.0-2.0*self.margin[1]))
    elif self.shape in ('circle','square'):
      l= min(self.width()*(1.0-2.0*self.margin[0]), self.height()*(1.0-2.0*self.margin[1]))
      rect= QtCore.QRect((self.width()-l)/2, (self.height()-l)/2, l, l)

    if self.shape in ('ellipse','circle'):
      painter.drawEllipse(rect)
    elif self.shape in ('rect','square'):
      painter.drawRect(rect)

    painter.restore()

    painter.setPen(self.palette().dark().color())
    painter.setBrush(QtCore.Qt.NoBrush)
    painter.drawRect(QtCore.QRect(0, 0, self.width() - 1, self.height() - 1))


class TSimplePanel(QtGui.QWidget):
  def __init__(self, title, size=(800,400), font_height_scale=100.0):
    QtGui.QWidget.__init__(self)
    self.font_height_scale= font_height_scale
    self.alignments= {
      '': QtCore.Qt.Alignment(),
      'left': QtCore.Qt.AlignLeft,
      'center': QtCore.Qt.AlignCenter,
      'right': QtCore.Qt.AlignRight,
      }
    self.size_policies= {
      'fixed': QtGui.QSizePolicy.Fixed,
      'minimum': QtGui.QSizePolicy.Minimum,
      'maximum': QtGui.QSizePolicy.Maximum,
      'preferred': QtGui.QSizePolicy.Preferred,
      'expanding': QtGui.QSizePolicy.Expanding,
      'minimum_expanding': QtGui.QSizePolicy.MinimumExpanding,
      'ignored': QtGui.QSizePolicy.Ignored,
      }
    self.param_common={
      'enabled': True,
      'font_size_range': (10,30),
      'minimum_size': None,  #None, or (horizontal_minimum_size, vertical_minimum_size)
      'maximum_size': None,  #None, or (horizontal_maximum_size, vertical_maximum_size)
      'size_policy': ('expanding', 'expanding'),  #(horizontal_size_policy, vertical_size_policy), or size_policy
      }
    self.widget_generator= {
      'duplicate': self.DuplicateWidget,
      'button': self.AddButton,
      'buttonchk': self.AddButtonCheckable,
      'combobox': self.AddComboBox,
      'lineedit': self.AddLineEdit,
      'radiobox': self.AddRadioBox,
      'sliderh': self.AddSliderH,
      'spacer': self.AddSpacer,
      'label': self.AddLabel,
      'textedit': self.AddTextEdit,
      'primitive_painer': self.AddPrimitivePainter,
      }
    self.resize(*size)  #window size
    self.setWindowTitle(title)
    self.widgets_in= {}
    self.widgets= {}
    self.layout_in= None
    self.layouts= {}

  #Add widgets from widget description dict.
  def AddWidgets(self, widgets):
    for name,(w_type, w_param) in widgets.iteritems():
      self.widgets_in[name]= (w_type, w_param)
    for name in widgets.iterkeys():
      w_type, w_param= self.widgets_in[name]
      self.widgets[name]= self.widget_generator[w_type](w_param)

  def Construct(self, layout):
    self.layout_in= layout
    self.setLayout(self.AddLayouts(self.layout_in))

    self.ResizeText(None)
    self.resizeEvent= self.ResizeText

    self.show()

  def ApplyCommonConfig(self, widget, param):
    widget.font_size_range= param['font_size_range']
    widget.setFont(QtGui.QFont('', widget.font_size_range[0]))
    if param['enabled'] is not None:  widget.setEnabled(param['enabled'])
    if param['minimum_size'] is not None:
      if param['minimum_size'][0]:  widget.setMinimumWidth(param['minimum_size'][0])
      if param['minimum_size'][1]:  widget.setMinimumHeight(param['minimum_size'][1])
    if param['maximum_size'] is not None:
      if param['maximum_size'][0]:  widget.setMaximumWidth(param['maximum_size'][0])
      if param['maximum_size'][1]:  widget.setMaximumHeight(param['maximum_size'][1])
    if param['size_policy'] is not None:
      if isinstance(param['size_policy'],str):  widget.setSizePolicy(self.size_policies[param['size_policy']])
      else:  widget.setSizePolicy(self.size_policies[param['size_policy'][0]], self.size_policies[param['size_policy'][1]])

  def ResizeTextOfObj(self, obj, font_size_range, size):
    font_size= min(font_size_range[1],max(font_size_range[0],int(size*font_size_range[0])))
    f= QtGui.QFont('', font_size)
    if isinstance(obj,QtGui.QLineEdit):
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
      if 'font_size_range' not in obj.__dict__:  continue
      self.ResizeTextOfObj(obj, obj.font_size_range, s)

  #Duplicate a widget (regenerate a widget with the same parameter).
  #NOTE: w_param is a widget name to be copied.
  def DuplicateWidget(self, w_param):
    src_widget_name= w_param
    w_type, w_param= self.widgets_in[src_widget_name]
    widget= self.widget_generator[w_type](w_param)
    return widget

  def AddButton(self, w_param):
    param= MergeDict2(copy.deepcopy(self.param_common), {
      'text': 'button',
      'onclick': None,
      }, w_param)
    btn= QtGui.QPushButton(param['text'], self)
    btn.setFocusPolicy(QtCore.Qt.NoFocus)
    #btn.setFlat(True)
    #btn.setToolTip('Click to make something happen')
    if param['onclick']:  btn.clicked.connect(lambda btn=btn: param['onclick'](self,btn))
    #btn.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
    btn.resize(btn.sizeHint())
    #btn.move(100, 150)
    self.ApplyCommonConfig(btn, param)
    return btn

  def AddButtonCheckable(self, w_param):
    param= MergeDict2(copy.deepcopy(self.param_common), {
      'text': ('button','button'),
      'checked': False,
      'onclick': None,
      }, w_param)
    btn= QtGui.QPushButton(param['text'][0], self)
    btn.setFocusPolicy(QtCore.Qt.NoFocus)
    btn.setCheckable(True)
    btn.setChecked(param['checked'])
    if param['onclick']:  btn.clicked.connect(lambda bnt=btn: (param['onclick'][0](self,btn) if param['onclick'][0] else None, btn.setText(param['text'][1])) if btn.isChecked() else (param['onclick'][1](self,btn) if param['onclick'][1] else None, btn.setText(param['text'][0])) )
    #btn.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
    btn.resize(btn.sizeHint())
    #btn.move(220, 100)
    self.ApplyCommonConfig(btn, param)
    return btn

  def AddComboBox(self, w_param):
    param= MergeDict2(copy.deepcopy(self.param_common), {
      'options': [],
      'index': 0,
      'onactivated': None,
      }, w_param)
    cmbbx= QtGui.QComboBox(self)
    cmbbx.setFocusPolicy(QtCore.Qt.NoFocus)
    for option in param['options']:
      cmbbx.addItem(option)
    if param['index'] is not None:  cmbbx.setCurrentIndex(param['index'])
    if param['onactivated']:  cmbbx.activated[str].connect(lambda _,cmbbx=cmbbx:param['onactivated'](self,cmbbx))
    #cmbbx.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
    cmbbx.resize(cmbbx.sizeHint())
    #cmbbx.move(10, 60)
    self.ApplyCommonConfig(cmbbx, param)
    return cmbbx

  def AddLineEdit(self, w_param):
    param= MergeDict2(copy.deepcopy(self.param_common), {
      'validator': None,  #'int'
      'size_policy': ('expanding', 'fixed'),
      }, w_param)
    edit= QtGui.QLineEdit(self)
    if param['validator']=='int':  edit.setValidator(QtGui.QIntValidator())
    edit.setMinimumHeight(10)
    edit.setMinimumWidth(10)
    #edit.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Fixed)
    edit.resize(edit.sizeHint())
    #edit.move(10, 60)
    self.ApplyCommonConfig(edit, param)
    return edit

  def AddRadioBox(self, w_param):
    param= MergeDict2(copy.deepcopy(self.param_common), {
      'options': [],
      'index': None,
      'layout': 'h',  #h(horizontal),v(vertical),grid
      'onclick': None,
      }, w_param)
    radiobox= TRadioBox(self)
    if param['onclick']:  clicked= lambda _,radiobox=radiobox:param['onclick'](self,radiobox)
    else:  clicked= None
    radiobox.Construct(param['layout'], param['options'], index=param['index'], onclick=clicked)
    self.ApplyCommonConfig(radiobox, param)
    return radiobox

  def AddSliderH(self, w_param):
    param= MergeDict2(copy.deepcopy(self.param_common), {
      'range': (0,10,1),
      'value': 0,
      'n_labels': 3,
      'slider_style': 0,
      'onvaluechange': None,
      }, w_param)
    slider= TSlider(self)
    if param['onvaluechange']:  onvaluechange= lambda _,slider=slider:param['onvaluechange'](self,slider)
    else:  onvaluechange= None
    slider.Construct(param['range'], n_labels=param['n_labels'], slider_style=param['slider_style'], onvaluechange=onvaluechange)
    if param['value'] is not None:  slider.setValue(param['value'])
    self.ApplyCommonConfig(slider, param)
    return slider

  def AddSpacer(self, w_param):
    param= MergeDict2(copy.deepcopy(self.param_common), {
      'w':1,
      'h':1,
      }, w_param)
    spacer= QtGui.QSpacerItem(param['w'], param['h'], QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
    #self.ApplyCommonConfig(spacer, param)
    return spacer

  def AddLabel(self, w_param):
    param= MergeDict2(copy.deepcopy(self.param_common), {
      'text':'',
      'selectable_by_mouse':False,  #Text is selectable by mouse.
      }, w_param)
    label= QtGui.QLabel(param['text'], self)
    #label.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
    if param['selectable_by_mouse']:
      label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
    self.ApplyCommonConfig(label, param)
    return label

  def AddTextEdit(self, w_param):
    param= MergeDict2(copy.deepcopy(self.param_common), {
      'text':'',
      'read_only':False,
      }, w_param)
    text= QtGui.QTextEdit(self)
    text.setText(param['text'])
    text.setReadOnly(param['read_only'])
    #text.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Preferred)
    self.ApplyCommonConfig(text, param)
    return text

  def AddPrimitivePainter(self, w_param):
    param= MergeDict2(copy.deepcopy(self.param_common), {
      'shape':'rect',
      'margin':(0.05,0.05),
      'color':(0,0,255),
      }, w_param)
    primitive= TPrimitivePainter(param['shape'],param['margin'],param['color'],self)
    self.ApplyCommonConfig(primitive, param)
    return primitive

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
      layout= QtGui.QGridLayout()
      for item_loc in items:
        if len(item_loc)==3:
          (item,r,c),rs,cs,align= item_loc,1,1,self.alignments['']
        elif len(item_loc)==4:
          (item,r,c),rs,cs,align= item_loc[:3],1,1,self.alignments[item_loc[3]]
        elif len(item_loc)==5:
          (item,r,c,rs,cs),align= item_loc,self.alignments['']
        elif len(item_loc)==6:
          item,r,c,rs,cs,align= item_loc
        else:
          raise Exception('Invalid grid item:',item_loc)
        if isinstance(item,str):
          widget= self.widgets[item]
          if isinstance(widget,QtGui.QSpacerItem):
            layout.addItem(widget,r,c,rs,cs,align)
          else:
            layout.addWidget(widget,r,c,rs,cs,align)
        else:
          sublayout= self.AddLayouts(item)
          layout.addLayout(sublayout,r,c,rs,cs,align)
    elif l_type=='tab':
      #As a tab is a widget in Qt, we make an outer layout with zero margin.
      #NOTE: The tab widget is stored into layout.tabs member variable.
      layout= QtGui.QVBoxLayout()
      layout.setContentsMargins(0, 0, 0, 0)
      layout.tabs= QtGui.QTabWidget()
      layout.addWidget(layout.tabs)
      layout.tab= []
      layout.tab_name_to_index= {}
      for tab_name,tab_layout in items:
        tab= QtGui.QWidget()
        layout.tab.append(tab)
        layout.tab_name_to_index[tab_name]= len(layout.tab)-1
        layout.tabs.addTab(tab,tab_name)
        sublayout= self.AddLayouts(tab_layout)
        tab.setLayout(sublayout)
      #For convenience, we define a setCurrentTab method to show a tab by name.
      layout.setCurrentTab= lambda tab_name:layout.tabs.setCurrentIndex(layout.tab_name_to_index[tab_name])

    self.layouts[name]= layout
    return layout


app= None

def InitPanelApp():
  global app
  app= QtGui.QApplication(sys.argv)
  return app

def RunPanelApp():
  global app
  sys.exit(app.exec_())


if __name__=='__main__':
  def Print(*s):
    for ss in s:  print ss,
    print ''

  widgets= {
    'btn1': (
      'button',{
        'text':'Close',
        'onclick':lambda w,obj:w.close()}),
    'btn2': (
      'buttonchk',{
        'text':('TurnOn','TurnOff'),
        'onclick': (lambda w,obj:Print('ON!'),
                    lambda w,obj:Print('OFF!'))}),
    'btn_totab10': (
      'button',{
        'text':'To tab1',
        'onclick':lambda w,obj:w.layouts['maintab'].setCurrentTab('tab1')}),
    'btn_totab11': ('duplicate', 'btn_totab10'),
    'btn_totab20': (
      'button',{
        'text':'To tab2',
        'onclick':lambda w,obj:w.layouts['maintab'].setCurrentTab('tab2')}),
    'btn_totab21': ('duplicate', 'btn_totab20'),
    'btn_totab30': (
      'button',{
        'text':'To tab3',
        'onclick':lambda w,obj:w.layouts['maintab'].setCurrentTab('tab3')}),
    'btn_totab31': ('duplicate', 'btn_totab30'),
    'cmb1': (
      'combobox',{
        'options':('Option-0','Option-1','Option-2','Other'),
        'index':1,
        'onactivated': lambda w,obj:(Print('Selected',obj.currentText()),
                                     w.widgets['edit_cmb1other'].setEnabled(obj.currentText()=='Other'))}),
    'edit_cmb1other': (
      'lineedit',{
        'validator':'int',
        'enabled':False}),
    'radbox1': (
      'radiobox',{
        'options':('Option-0','Option-1','Option-2','Other'),
        'layout': 'h',
        'index': 0,
        'onclick': lambda w,obj:(Print('Selected',obj.group.checkedButton().text()),
                                 w.widgets['edit_radbox1other'].setEnabled(obj.group.checkedButton().text()=='Other'))}),
    'edit_radbox1other': (
      'lineedit',{
        'validator':'int',
        'enabled':False}),
    'radbox2': (
      'radiobox',{
        'options':('Option-A','Option-B','Option-C','Other'),
        'layout': 'v',
        'index': None,
        'onclick': lambda w,obj:(Print('Selected',obj.group.checkedButton().text()),
                                 w.widgets['slider_radbox2other'].setEnabled(obj.group.checkedButton().text()=='Other'))}),
    'slider_radbox2other': (
      'sliderh',{
        'range': (1000,1800,100),
        'value': 1600,
        'n_labels': 3,
        'slider_style':1,
        'enabled':False,
        'onvaluechange': lambda w,obj:Print('Value:',obj.value())}),
    'label_tab2': (
      'label',{
        'text': 'Select a tab to go',
        'size_policy': ('expanding', 'minimum')}),
    'textedit1': (
      'textedit',{
        'text': 'Example\nOf\nTextEdit',}),
    'primitive_painer1': (
      'primitive_painer',{
        'color': (0,0,255),
        'margin': (0,0),
        'minimum_size': (None,20),
        'maximum_size': (None,20),
        'size_policy': ('expanding', 'fixed')}),
    'spacer1': ('spacer', {}),
    }
  #Layout option: vbox, hbox, grid, tab
  #layout= ('boxv',None,(
             #('grid',None, (('btn1',0,0),('btn2',0,1,1,2),
                            #('spacer1',1,0),('cmb1',1,1),('edit_cmb1other',1,2)) ),
             #('boxh',None, ('radbox1','edit_radbox1other') ),
             #('boxv',None, ('radbox2','slider_radbox2other') ),
             #'spacer1',
           #))
  layout= ('tab','maintab',(
           ('tab1',
              ('boxv',None,
                (
                  ('grid',None, (('btn1',0,0),('btn2',0,1,1,2),
                                  ('spacer1',1,0),('cmb1',1,1),('edit_cmb1other',1,2)) ),
                  ('boxh',None, ('radbox1','edit_radbox1other') ),
                  ('boxv',None, ('radbox2','slider_radbox2other') ),
                  ('boxh',None, ('btn_totab20', 'btn_totab30') ),
                  'spacer1',
                )) ),
            ('tab2', ('boxv',None, ('label_tab2', 'btn_totab10', 'btn_totab31') ) ),
            ('tab3', ('boxv',None, ('primitive_painer1','btn_totab11', 'btn_totab21', 'textedit1') ) ),
            ))

  InitPanelApp()
  panel= TSimplePanel('Simple Panel Example', size=(600,400), font_height_scale=300.0)
  panel.AddWidgets(widgets)
  panel.Construct(layout)
  RunPanelApp()

