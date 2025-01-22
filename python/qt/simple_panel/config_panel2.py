#!/usr/bin/python3
#\file    config_panel2.py
#\brief   Test of a parameter configuration panel.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.01, 2022

import sys
import yaml
#from PyQt4 import QtCore,QtGui
from _import_qt import *
#from slider4 import TSlider
from simple_panel1 import TSimplePanel,InitPanelApp,RunPanelApp

if __name__=='__main__':
  def Print(*s):
    for ss in s:  print(ss, end=' ')
    print('')

  params={
    'sensitivity_sl': 0.08,
    'sensitivity_oc':0.2,
    'sensitivity_oo':0.5,
    'area_ratio': 0.3,
    }
  def UpdateParam(name, value):
    params[name]= value
    print(yaml.dump(params))

  def AddConfigSliderWidget(widgets, name, prange):
    widgets['slider_{}'.format(name)]= (
      'sliderh',{
        'range': prange,
        'value': params[name],
        'n_labels': 3,
        'slider_style':1,
        'onvaluechange': lambda w,obj:UpdateParam(name,obj.value())} )
    widgets['label_{}'.format(name)]= (
      'label',{
        'text': name,
        'size_policy': ('minimum', 'minimum')} )
  widgets= {
    'btn_exit': (
      'button',{
        'text':'Exit',
        'size_policy': ('expanding', 'minimum'),
        'onclick':lambda w,obj:w.close()}),
    }
  AddConfigSliderWidget(widgets, 'sensitivity_sl', (0.0,0.5,0.01))
  AddConfigSliderWidget(widgets, 'sensitivity_oc', (0.0,2.0,0.1))
  AddConfigSliderWidget(widgets, 'sensitivity_oo', (0.0,2.0,0.1))
  AddConfigSliderWidget(widgets, 'area_ratio', (0.0,2.0,0.1))
  layout= ('boxv',None,
            (
              ('boxh',None, ('label_sensitivity_sl','slider_sensitivity_sl') ),
              ('boxh',None, ('label_sensitivity_oc','slider_sensitivity_oc') ),
              ('boxh',None, ('label_sensitivity_oo','slider_sensitivity_oo') ),
              ('boxh',None, ('label_area_ratio','slider_area_ratio') ),
              'btn_exit',
            ))

  InitPanelApp()
  panel= TSimplePanel('Config Panel', size=(400,300), font_height_scale=300.0)
  panel.AddWidgets(widgets)
  panel.Construct(layout)
  RunPanelApp()
