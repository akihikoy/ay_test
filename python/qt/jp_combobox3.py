#!/usr/bin/python
# -*- coding: utf-8 -*-
#\file    jp_combobox3.py
#\brief   Test of ComboBox with Japanese.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Mar.02, 2023

import sys
from PyQt4 import QtCore,QtGui

def Print(*s):
  for ss in s:
    if isinstance(ss,QtCore.QString):  print ss.toUtf8(),
    else:  print ss
  print ''

class TComboBox(QtGui.QWidget):
  def __init__(self):
    QtGui.QWidget.__init__(self)
    self.InitUI()

  def InitUI(self):
    # Set window size.
    self.resize(320, 240)

    # Set window title
    self.setWindowTitle(QtCore.QString.fromUtf8('コンボボックス'))

    # Add a button
    btn1= QtGui.QPushButton(QtCore.QString.fromUtf8('_________終了？_________'), self)
    #btn1.setFlat(True)
    btn1.setToolTip(QtCore.QString.fromUtf8('何かを起こしたければクリックしたまえ'))
    btn1.clicked.connect(lambda:self.close() if self.cmbbx1.currentText()==QtCore.QString.fromUtf8('オプション-2') or (self.cmbbx1.currentText()==QtCore.QString.fromUtf8('その他') and self.edit1.text()==QtCore.QString.fromUtf8('終了')) else Print(QtCore.QString.fromUtf8('ヒント：終了したければオプション-2を選択するか，その他を選択し終了と入力せよ')))
    btn1.resize(btn1.sizeHint())
    btn1.move(100, 150)
    self.btn1= btn1

    cmbbx1= QtGui.QComboBox(self)
    cmbbx1.addItem(QtCore.QString.fromUtf8('オプション-1'))
    cmbbx1.addItem(QtCore.QString.fromUtf8('オプション-2'))
    cmbbx1.addItem(QtCore.QString.fromUtf8('オプション-3'))
    cmbbx1.addItem(QtCore.QString.fromUtf8('その他'))
    cmbbx1.setCurrentIndex(1)
    cmbbx1.move(10, 60)
    cmbbx1.activated[str].connect(lambda:(self.edit1.setEnabled(self.cmbbx1.currentText()==QtCore.QString.fromUtf8('その他')), Print(QtCore.QString.fromUtf8('選択された'),self.cmbbx1.currentText())))
    self.cmbbx1= cmbbx1

    edit1= QtGui.QLineEdit(self)
    #edit1.setValidator(QtGui.QIntValidator())
    self.edit1= edit1
    edit1.move(120, 60)
    edit1.resize(90, 30)
    self.edit1.setEnabled(self.cmbbx1.currentText()==QtCore.QString.fromUtf8('その他'))

    # Show window
    self.show()

# Create an PyQT4 application object.
a = QtGui.QApplication(sys.argv)

# The QWidget widget is the base class of all user interface objects in PyQt4.
w = TComboBox()

sys.exit(a.exec_())