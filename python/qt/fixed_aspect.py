#!/usr/bin/python3
#\file    fixed_aspect.py
#\brief   Fixed aspect ratio example.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Apr.15, 2021

#src: https://wiki.python.org/moin/PyQt/Creating%20a%20widget%20with%20a%20fixed%20aspect%20ratio

import sys
#from PyQt4.QtCore import pyqtSignal, QSize, Qt
#from PyQt4.QtGui import *
from _import_qt import *

class MyWidget(QtGui.QWidget):

    clicked = QtCore.pyqtSignal()
    keyPressed = QtCore.pyqtSignal(str)

    def __init__(self, parent = None):

        QtGui.QWidget.__init__(self, parent)
        self.color= QtGui.QColor(0, 0, 0)
        sizePolicy= QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHeightForWidth(True)
        self.setSizePolicy(sizePolicy)

    def paintEvent(self, event):

        painter = QtGui.QPainter()
        painter.begin(self)
        painter.fillRect(event.rect(), QtGui.QBrush(self.color))
        painter.end()

    def keyPressEvent(self, event):

        self.keyPressed.emit(event.text())
        event.accept()

    def mousePressEvent(self, event):

        self.setFocus(Qt.OtherFocusReason)
        event.accept()

    def mouseReleaseEvent(self, event):

        if event.button() == Qt.LeftButton:

            self.color = QtGui.QColor(self.color.green(), self.color.blue(),
                                127 - self.color.red())
            self.update()
            self.clicked.emit()
            event.accept()

    def sizeHint(self):

        return QtCore.QSize(400, 600)

    def heightForWidth(self, width):

        return int(width * 1.5)


if __name__ == "__main__":

    app = QtGui.QApplication(sys.argv)
    window = QtGui.QWidget()

    mywidget = MyWidget()
    label = QtGui.QLabel()

    mywidget.clicked.connect(label.clear)
    mywidget.keyPressed.connect(label.setText)

    layout = QtGui.QVBoxLayout()
    layout.addWidget(mywidget, 0, QtCore.Qt.AlignCenter)
    layout.addWidget(label)
    window.setLayout(layout)

    window.show()
    sys.exit(app.exec_())
