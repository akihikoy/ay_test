#!/usr/bin/python
#\file    _import_qt.py
#\brief   Import PyQt modules;
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jan.22, 2025

#Using PyQt4:
#from PyQt4 import QtGui,QtCore,QtTest,uic
#from PyQt4.QtCore import pyqtSlot
#from PyQt4.QtGui import *

#Using PyQt5:

#Quick solution to use PyQt4 program with PyQt5.
from PyQt5 import QtCore,QtWidgets,QtTest
import PyQt5.QtGui as PyQt5QtGui
QtGui= QtWidgets
for component in ('QFont', 'QPalette', 'QColor', 'QLinearGradient', 'QPainter', 'QImage', 'QPolygon', 'QPen', 'QBrush', 'QPixmap', 'QPainterPath', 'QIntValidator'):
  setattr(QtGui,component, getattr(PyQt5QtGui,component))


