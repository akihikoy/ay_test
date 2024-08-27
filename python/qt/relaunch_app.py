#!/usr/bin/python
#\file    relaunch_app.py
#\brief   Test of relaunching the Qt app.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.28, 2024
import sys
from PyQt4 import QtCore, QtGui
import os

def RelaunchProgram():
  print("Relaunching program...")
  python= sys.executable
  os.execl(python, python, *sys.argv)

class TRelaunchBox(QtGui.QWidget):
  def __init__(self):
    QtGui.QWidget.__init__(self)
    self.InitUI()
    self.flag_relaunch= False

  def CloseAndRelaunch(self):
    self.flag_relaunch= True
    self.close()

  def InitUI(self):
    # Set window size.
    self.resize(320, 280)

    # Set window title
    self.setWindowTitle("Program relaunch test")

    btn_exit= QtGui.QPushButton('Exit', self)
    btn_exit.setToolTip('Click to exit')
    btn_exit.resize(btn_exit.sizeHint())
    btn_exit.clicked.connect(lambda b: self.close())
    btn_exit.move(100, 40)

    btn_relaunch1= QtGui.QPushButton('Relaunch the program', self)
    btn_relaunch1.setToolTip('Click to relaunch')
    btn_relaunch1.resize(btn_relaunch1.sizeHint())
    btn_relaunch1.clicked.connect(lambda b: RelaunchProgram())
    btn_relaunch1.move(100, 120)

    btn_relaunch2= QtGui.QPushButton('Relaunch after GUI exit', self)
    btn_relaunch2.setToolTip('Click to relaunch')
    btn_relaunch2.resize(btn_relaunch2.sizeHint())
    btn_relaunch2.clicked.connect(lambda b: self.CloseAndRelaunch())
    btn_relaunch2.move(100, 200)

    # Show window
    self.show()

if __name__=='__main__':
  a = QtGui.QApplication(sys.argv)
  w = TRelaunchBox()
  #sys.exit(a.exec_())
  exit_code= a.exec_()

  print 'The GUI app has ended with exit code: {}.'.format(exit_code)

  if w.flag_relaunch:
    RelaunchProgram()
