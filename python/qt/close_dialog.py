#!/usr/bin/python
#\file    close_dialog.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Apr.01, 2017

# http://stackoverflow.com/questions/14834494/pyqt-clicking-x-doesnt-trigger-closeevent

import sys
from PyQt4 import QtGui, QtCore, uic

class MainWindow(QtGui.QWidget):
  def __init__(self):
    QtGui.QWidget.__init__(self)

    # Set window size.
    self.resize(320, 240)

    # Set window title
    self.setWindowTitle("Hello World!")

    # Add a button
    btn= QtGui.QPushButton('Hello World!', self)
    btn.setToolTip('Click to quit!')
    btn.clicked.connect(self.close)
    btn.resize(btn.sizeHint())
    btn.move(100, 80)

  def closeEvent(self, event):
    print("event")
    reply = QtGui.QMessageBox.question(self, 'Message',
        "Are you sure to quit?", QtGui.QMessageBox.Yes, QtGui.QMessageBox.No)

    if reply == QtGui.QMessageBox.Yes:
      event.accept()
    else:
      event.ignore()


if __name__ == "__main__":
  app = QtGui.QApplication(sys.argv)
  win = MainWindow()
  win.show()
  sys.exit(app.exec_())

