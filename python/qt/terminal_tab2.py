#!/usr/bin/python
#\file    terminal_tab2.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Mar.31, 2017

import sys
from PyQt4 import QtCore,QtGui

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

    E= 'Enter'
    terminals= [
      ('s1',[
        ('rviz',lambda:self.SendCmd('s1','ros',E,'bxmaster',E,'rviz',E)),
        ('kill',lambda:self.SendCmd('s1','C-c'))  ]),
      ('s2',[
        ('ls',lambda:self.SendCmd('s2','ls',E)),
        ('nodes',lambda:self.SendCmd('s1','ros',E,'bxmaster',E,'rostopic list',E)),
        ('topics',lambda:self.SendCmd('s1','ros',E,'bxmaster',E,'rosnode list',E)) ]),
      ('e',[
        ('Exit',self.close) ])
      ]
    self.Terminals= terminals
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
    for r,(term,row) in enumerate(self.Terminals):
      #label= QtGui.QLabel()
      #label.setText(term)
      #grid.addWidget(label, r, 0)
      btn0= QtGui.QPushButton('({term})'.format(term=term))
      btn0.setFlat(True)
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
    for r,(term,row) in enumerate(self.Terminals):
      tab= QtGui.QWidget()
      tabs.addTab(tab, term)

      terminal= QtGui.QWidget(self)
      hBoxlayout= QtGui.QHBoxLayout()
      #tab.resize(600, 200)
      tab.setLayout(hBoxlayout)
      hBoxlayout.addWidget(terminal)
      self.qtterm[term]= terminal
      #self.StartProc(
          #'urxvt',
          #['-embed', str(terminal.winId()),
            #'-e', 'tmux', 'new', '-s', term])

    return tabs

  def ShowTermTab(self, term):
    self.qttabs.setCurrentIndex(self.term_to_idx[term])

  def StartProc(self, prog, args):
      child= QtCore.QProcess()
      self.Processes.append(child)
      child.start(prog, args)

  def SendCmd(self, term, *cmd):
    self.ShowTermTab(term)
    self.StartProc('tmux', ['send-keys', '-t', term+':0'] + list(cmd))

  def CreateTerminals(self):
    for r,(term,row) in enumerate(self.Terminals):
      self.qttabs.setCurrentIndex(r)
      self.StartProc(
          'urxvt',
          ['-embed', str(self.qtterm[term].winId()),
            '-e', 'tmux', 'new', '-s', term])
    self.qttabs.setCurrentIndex(0)

  def Exit(self):
    #for p in self.Processes[len(self.Terminals):]:
      #print p.pid()
    for r,(term,row) in enumerate(self.Terminals):
      self.StartProc('tmux', ['send-keys', '-t', term+':0', 'C-c'])
      self.StartProc('tmux', ['send-keys', '-t', term+':0', 'exit', 'Enter'])
    #sys.exit()

  # Override closing event
  def closeEvent(self, event):
    quit_msg= 'Really exit the program?'
    reply= QtGui.QMessageBox.question(self, 'Message',
                     quit_msg, QtGui.QMessageBox.Yes, QtGui.QMessageBox.No)
    if reply==QtGui.QMessageBox.Yes:
      self.Exit()
      event.accept()
    else:
      event.ignore()

# Create an PyQT4 application object.
app= QtGui.QApplication(sys.argv)

win= TTerminalTab()

sys.exit(app.exec_())
