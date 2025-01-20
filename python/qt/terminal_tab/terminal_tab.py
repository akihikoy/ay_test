#!/usr/bin/python3
#\file    terminal_tab.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Mar.31, 2017

import sys
from PyQt4 import QtCore,QtGui

def Print(s):
  print(s)

class TTerminalTab(QtGui.QWidget):
  def __init__(self):
    QtGui.QWidget.__init__(self)
    self.InitUI()

  def InitUI(self):
    # Set window size.
    self.resize(800, 400)
    self.Processes= []

    # Set window title
    self.setWindowTitle("Terminals in tabs")

    terminals= [('session1', [('a0',lambda:Print('0')), ('a1',lambda:Print('1'))] ),
                ('session2', [('b0',self.Test), ('b1',lambda:Print('y')), ('b2',lambda:Print('z'))] ),
                ('session3', [('c0',self.Exit)] )]
    self.terminals= terminals
    self.term_to_idx= {term:r for r,(term,row) in enumerate(terminals)}

    # Horizontal box layout
    hBoxlayout= QtGui.QHBoxLayout()
    #hBoxlayout.addStretch()
    self.setLayout(hBoxlayout)
    #hBoxlayout.addStretch()

    self.qttabs= self.MakeTabs()
    hBoxlayout.addWidget(self.qttabs)

    # Grid layout
    grid= QtGui.QGridLayout()
    wg= QtGui.QWidget()
    wg.setLayout(grid)
    hBoxlayout.addWidget(wg)

    # Add buttons on grid
    for r,(term,row) in enumerate(self.terminals):
      #label= QtGui.QLabel()
      #label.setText(term)
      #grid.addWidget(label, r, 0)
      btn0= QtGui.QPushButton(term)
      btn0.clicked.connect(lambda clicked,term=term:self.ShowTermTab(term))
      grid.addWidget(btn0, r, 0)
      for c,(name,f) in enumerate(row):
        btn= QtGui.QPushButton(name)
        btn.clicked.connect(f)
        grid.addWidget(btn, r, 1+c)

    # Show window
    self.show()
    self.CreateTerminals()

  def MakeTabs(self):
    tabs= QtGui.QTabWidget()

    # Resize width and height
    #tabs.resize(600, 400)

    self.qtterm= {}
    for r,(term,row) in enumerate(self.terminals):
      tab= QtGui.QWidget()
      tabs.addTab(tab, term)

      terminal= QtGui.QWidget(self)
      hBoxlayout= QtGui.QHBoxLayout()
      #tab.resize(600, 200)
      tab.setLayout(hBoxlayout)
      hBoxlayout.addWidget(terminal)
      self.qtterm[term]= terminal
      #self.RunCmd(
          #'urxvt',
          #['-embed', str(terminal.winId()),
            #'-e', 'tmux', 'new', '-s', term])

    return tabs

  def ShowTermTab(self, term):
    self.qttabs.setCurrentIndex(self.term_to_idx[term])

  def RunCmd(self, prog, args):
      child= QtCore.QProcess()
      self.Processes.append(child)
      child.start(prog, args)

  def CreateTerminals(self):
    for r,(term,row) in enumerate(self.terminals):
      self.qttabs.setCurrentIndex(r)
      self.RunCmd(
          'urxvt',
          ['-embed', str(self.qtterm[term].winId()),
            '-e', 'tmux', 'new', '-s', term])
    self.qttabs.setCurrentIndex(0)

  def Test(self):
    for r,(term,row) in enumerate(self.terminals):
      self.RunCmd(
          'tmux', ['send-keys', '-t', term+':0', 'ls', 'Enter'])

  def Exit(self):
    for r,(term,row) in enumerate(self.terminals):
      self.RunCmd(
          'tmux', ['send-keys', '-t', term+':0', 'exit', 'Enter'])
    sys.exit()

# Create an PyQT4 application object.
a = QtGui.QApplication(sys.argv)

w = TTerminalTab()

sys.exit(a.exec_())
