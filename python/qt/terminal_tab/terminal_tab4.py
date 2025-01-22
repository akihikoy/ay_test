#!/usr/bin/python3
#\file    terminal_tab4.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Apr.14, 2017

import sys
#from PyQt4 import QtCore,QtGui,QtTest
from _import_qt import *

class TTerminalTab(QtGui.QWidget):
  def __init__(self):
    QtGui.QWidget.__init__(self)
    self.InitUI()

  def InitUI(self):
    # Set window size.
    self.resize(800, 400)
    self.Processes= []

    # Set window title
    self.setWindowTitle("Baxter Launcher")

    E= 'Enter'
    terminals= [
      ('main1',[
        ('Init',lambda:self.SendCmdToAll(*self.InitCommand)),
        ('Exit',self.close) ]),
      ('s1',[
        ('pair',('rviz',lambda:self.SendCmd('s1','rviz',E)),
        ('kill',lambda:self.SendCmd('s1','C-c')))  ]),
      ('s2',[
        ('ls',lambda:self.SendCmd('s2','ls',E)),
        ('nodes',lambda:self.SendCmd('s1','rostopic list',E)),
        ('topics',lambda:self.SendCmd('s1','rosnode list',E)) ]),
      ]
    self.InitCommand= ['ros',E,'bxmaster',E]
    self.ExitCommand= [E,'C-c']
    self.Terminals= terminals
    self.term_to_idx= {term:r for r,(term,row) in enumerate(terminals)}

    # Horizontal box layout
    hBoxlayout= QtGui.QHBoxLayout()
    self.setLayout(hBoxlayout)

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
      for c,commands in enumerate(row):
        if commands[0]=='pair':
          name1,f1= commands[1]
          name2,f2= commands[2]
          btn= QtGui.QPushButton(name1)
          btn.setCheckable(True)
          btn.clicked.connect(lambda b,btn=btn,name1=name1,f1=f1,name2=name2,f2=f2:
                                (f1(),btn.setText(name2)) if btn.isChecked() else (f2(),btn.setText(name1)))
          grid.addWidget(btn, r, 1+c)
        else:
          name,f= commands
          btn= QtGui.QPushButton(name)
          btn.clicked.connect(f)
          grid.addWidget(btn, r, 1+c)

    # Show window
    self.show()
    self.CreateTerminals()

  def MakeTabs(self):
    tabs= QtGui.QTabWidget()

    self.qtterm= {}
    for r,(term,row) in enumerate(self.Terminals):
      tab= QtGui.QWidget()
      tabs.addTab(tab, term)

      terminal= QtGui.QWidget(self)
      hBoxlayout= QtGui.QHBoxLayout()
      tab.setLayout(hBoxlayout)
      hBoxlayout.addWidget(terminal)
      self.qtterm[term]= terminal

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

  def SendCmdToAll(self, *cmd):
    for r,(term,row) in enumerate(self.Terminals):
      self.ShowTermTab(term)
      self.StartProc('tmux', ['send-keys', '-t', term+':0'] + list(cmd))

  def CreateTerminals(self):
    for r,(term,row) in enumerate(self.Terminals):
      self.qttabs.setCurrentIndex(r)
      self.StartProc(
          'urxvt',
          ['-embed', str(self.qtterm[term].winId()),
            '-e', 'tmux', 'new', '-s', term])
      QtTest.QTest.qWait(100)
    #QtTest.QTest.qWait(200*len(self.Terminals))
    #for r,(term,row) in enumerate(self.Terminals):
      #self.StartProc('tmux', ['send-keys', '-t', term+':0'] + self.InitCommand)
    self.qttabs.setCurrentIndex(0)

  def Exit(self):
    for r,(term,row) in enumerate(self.Terminals):
      self.qttabs.setCurrentIndex(r)
      self.StartProc('tmux', ['send-keys', '-t', term+':0'] + self.ExitCommand)
      QtTest.QTest.qWait(200)
      self.StartProc('tmux', ['send-keys', '-t', term+':0', 'exit', 'Enter'])
      QtTest.QTest.qWait(100)

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

if __name__=='__main__':
  app= QtGui.QApplication(sys.argv)
  win= TTerminalTab()
  sys.exit(app.exec_())
