#!/usr/bin/python3
#\file    config_panel1.py
#\brief   Test of a parameter configuration panel.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jul.30, 2022

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

  widgets= {
    'slider_sensitivity_sl': (
      'sliderh',{
        'range': (0.0,0.5,0.01),
        'value': params['sensitivity_sl'],
        'n_labels': 3,
        'slider_style':1,
        'onvaluechange': lambda w,obj:UpdateParam('sensitivity_sl',obj.value())}),
    'label_sensitivity_sl': (
      'label',{
        'text': 'sensitivity_sl',
        'size_policy': ('minimum', 'minimum')}),
    'slider_sensitivity_oc': (
      'sliderh',{
        'range': (0.0,2.0,0.1),
        'value': params['sensitivity_oc'],
        'n_labels': 3,
        'slider_style':1,
        'onvaluechange': lambda w,obj:UpdateParam('sensitivity_oc',obj.value())}),
    'label_sensitivity_oc': (
      'label',{
        'text': 'sensitivity_oc',
        'size_policy': ('minimum', 'minimum')}),
    'slider_sensitivity_oo': (
      'sliderh',{
        'range': (0.0,2.0,0.1),
        'value': params['sensitivity_oo'],
        'n_labels': 3,
        'slider_style':1,
        'onvaluechange': lambda w,obj:UpdateParam('sensitivity_oo',obj.value())}),
    'label_sensitivity_oo': (
      'label',{
        'text': 'sensitivity_oo',
        'size_policy': ('minimum', 'minimum')}),
    'slider_area_ratio': (
      'sliderh',{
        'range': (0.0,2.0,0.1),
        'value': params['area_ratio'],
        'n_labels': 3,
        'slider_style':1,
        'onvaluechange': lambda w,obj:UpdateParam('area_ratio',obj.value())}),
    'label_area_ratio': (
      'label',{
        'text': 'area_ratio',
        'size_policy': ('minimum', 'minimum')}),
    'btn_exit': (
      'button',{
        'text':'Exit',
        'size_policy': ('expanding', 'minimum'),
        'onclick':lambda w,obj:w.close()}),
    }
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
