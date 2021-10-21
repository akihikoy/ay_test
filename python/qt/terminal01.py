#!/usr/bin/python
#\file    terminal01.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Feb.07, 2017

#NOTE: need to install: tmux rxvt-unicode-256color

import sys,os
from PyQt4.QtCore import *
from PyQt4.QtGui import *

class embeddedTerminal(QWidget):

    def __init__(self):
        self.pid= str(os.getpid())
        QWidget.__init__(self)
        self._processes = []
        self.resize(600, 600)
        layout= QVBoxLayout(self)
        self.terminal1 = QWidget(self)
        layout.addWidget(self.terminal1)
        self._start_process(
            #'xterm',
            #['-into', str(self.terminal1.winId()),
            'urxvt',
            ['-embed', str(self.terminal1.winId()),
             '-e', 'tmux', 'new', '-s', self.pid+'-session1'])
        self.terminal2 = QWidget(self)
        layout.addWidget(self.terminal2)
        self._start_process(
            #'xterm',
            #['-into', str(self.terminal2.winId()),
            'urxvt',
            ['-embed', str(self.terminal2.winId()),
             '-e', 'tmux', 'new', '-s', self.pid+'-session2'])
        button = QPushButton('List files')
        layout.addWidget(button)
        button.clicked.connect(self._list_files)
        button2 = QPushButton('Exit')
        layout.addWidget(button2)
        button2.clicked.connect(self._exit)

    #def __del__(self):
        #self._start_process(
            #'tmux', ['send-keys', '-t', 'session1:0', 'exit', 'Enter'])

    def _start_process(self, prog, args):
        child = QProcess()
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
    app = QApplication(sys.argv)
    main = embeddedTerminal()
    main.show()
    sys.exit(app.exec_())

