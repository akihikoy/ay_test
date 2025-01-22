#!/usr/bin/python3
#\file    terminal01.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Feb.07, 2017

#NOTE: need to install: tmux rxvt-unicode-256color

import sys,os
#from PyQt4.QtCore import *
#from PyQt4.QtGui import *
from _import_qt import *

class embeddedTerminal(QtGui.QWidget):

    def __init__(self):
        self.pid= str(os.getpid())
        QtGui.QWidget.__init__(self)
        self._processes = []
        self.resize(600, 600)
        layout= QtGui.QVBoxLayout(self)
        self.terminal1 = QtGui.QWidget(self)
        self.terminal1.setAttribute(QtCore.Qt.WA_NativeWindow)
        self.terminal1.resize(500, 250)
        self.terminal1.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        layout.addWidget(self.terminal1)
        self._start_process(
            #'xterm',
            #['-into', str(int(self.terminal1.winId())),
            'urxvt',
            ['-embed', str(int(self.terminal1.winId())),
             '-e', 'tmux', 'new', '-s', self.pid+'-session1'])
        self.terminal2 = QtGui.QWidget(self)
        self.terminal2.setAttribute(QtCore.Qt.WA_NativeWindow)
        self.terminal2.resize(500, 250)
        self.terminal2.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        layout.addWidget(self.terminal2)
        self._start_process(
            #'xterm',
            #['-into', str(int(self.terminal2.winId())),
            'urxvt',
            ['-embed', str(int(self.terminal2.winId())),
             '-e', 'tmux', 'new', '-s', self.pid+'-session2'])
        button = QtGui.QPushButton('List files')
        layout.addWidget(button)
        button.clicked.connect(self._list_files)
        button2 = QtGui.QPushButton('Exit')
        layout.addWidget(button2)
        button2.clicked.connect(self._exit)

    #def __del__(self):
        #self._start_process(
            #'tmux', ['send-keys', '-t', 'session1:0', 'exit', 'Enter'])

    def _start_process(self, prog, args):
        child = QtCore.QProcess()
        self._processes.append(child)
        child.start(prog, args)

    def _list_files(self):
        self._start_process(
            'tmux', ['send-keys', '-t', self.pid+'-session1:0', 'ls', 'Enter'])
        self._start_process(
            'tmux', ['send-keys', '-t', self.pid+'-session2:0', 'ls /', 'Enter'])

    def _exit(self):
        self._start_process(
            'tmux', ['send-keys', '-t', self.pid+'-session1:0', 'exit', 'Enter'])
        self._start_process(
            'tmux', ['send-keys', '-t', self.pid+'-session2:0', 'exit', 'Enter'])

if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    main = embeddedTerminal()
    main.show()
    sys.exit(app.exec_())

