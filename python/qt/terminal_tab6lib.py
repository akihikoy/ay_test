#!/usr/bin/python
#\file    terminal_tab5lib.py
#\brief   Simple Tab-Terminal GUI command launcher (library).
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Sep.10, 2017
#Requirements: tmux rxvt-unicode-256color

import sys
import signal
from PyQt4 import QtCore,QtGui,QtTest

class TTerminalTab(QtGui.QWidget):
  def __init__(self,title,terminals,exit_command):
    QtGui.QWidget.__init__(self)
    self.InitUI(title,terminals,exit_command)

  def CmdToLambda(self,term,cmd):
    if cmd==':close':  return self.close
    if len(cmd)==0:  return lambda:None
    if cmd[0]==':all':  return lambda:self.SendCmdToAll(cmd[1:])
    return lambda:self.SendCmd(term,cmd)

  def InitUI(self,title,terminals,exit_command):
    # Set window size.
    self.resize(800, 400)
    self.Processes= []

    # Set window title
    self.WinTitle= title
    self.setWindowTitle(self.WinTitle)

    self.ExitCommand= exit_command
    self.Terminals= terminals
    self.term_to_idx= {term:r for r,(term,row) in enumerate(self.Terminals)}

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
      btn0.setFocusPolicy(QtCore.Qt.NoFocus)
      btn0.clicked.connect(lambda clicked,term=term:self.ShowTermTab(term))
      grid.addWidget(btn0, r, 0)
      for c,commands in enumerate(row):
        if commands[0]==':pair':
          name1,f1= commands[1][0],self.CmdToLambda(term,commands[1][1])
          name2,f2= commands[2][0],self.CmdToLambda(term,commands[2][1])
          btn= QtGui.QPushButton(name1)
          btn.setCheckable(True)
          btn.setFocusPolicy(QtCore.Qt.NoFocus)
          btn.clicked.connect(lambda b,btn=btn,name1=name1,f1=f1,name2=name2,f2=f2:
                                (f1(),btn.setText(name2)) if btn.isChecked() else (f2(),btn.setText(name1)))
          grid.addWidget(btn, r, 1+c)
        else:
          name,f= commands[0],self.CmdToLambda(term,commands[1])
          btn= QtGui.QPushButton(name)
          btn.setFocusPolicy(QtCore.Qt.NoFocus)
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

  def SendCmd(self, term, cmd):
    self.ShowTermTab(term)
    self.StartProc('tmux', ['send-keys', '-t', term+':0'] + list(cmd))

  def SendCmdToAll(self, cmd):
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
      #QtTest.QTest.qWait(100)
    QtTest.QTest.qWait(200)
    #QtTest.QTest.qWait(200*len(self.Terminals))
    #for r,(term,row) in enumerate(self.Terminals):
      #self.StartProc('tmux', ['send-keys', '-t', term+':0'] + self.InitCommand)
    self.qttabs.setCurrentIndex(0)

  def Exit(self):
    for r,(term,row) in enumerate(self.Terminals):
      self.qttabs.setCurrentIndex(r)
      self.StartProc('tmux', ['send-keys', '-t', term+':0'] + self.ExitCommand)
    QtTest.QTest.qWait(200)
    for r,(term,row) in enumerate(self.Terminals):
      self.qttabs.setCurrentIndex(r)
      self.StartProc('tmux', ['send-keys', '-t', term+':0', 'exit', 'Enter'])
    QtTest.QTest.qWait(100)

  # Override closing event
  def closeEvent(self, event):
    quit_msg= 'Really exit {title}?'.format(title=self.WinTitle)
    reply= QtGui.QMessageBox.question(self, 'Message',
                     quit_msg, QtGui.QMessageBox.Yes, QtGui.QMessageBox.No)
    if reply==QtGui.QMessageBox.Yes:
      self.Exit()
      event.accept()
    else:
      event.ignore()

def RunTerminalTab(title,terminals,exit_command):
  app= QtGui.QApplication(sys.argv)
  win= TTerminalTab(title,terminals,exit_command)
  signal.signal(signal.SIGINT, lambda signum,frame,win=win: (win.Exit(),QtGui.QApplication.quit()) )
  timer= QtCore.QTimer()
  timer.start(500)  # You may change this if you wish.
  timer.timeout.connect(lambda: None)  # Let the interpreter run each 500 ms.
  sys.exit(app.exec_())

if __name__=='__main__':
  E= 'Enter'
  terminals= [
    ('main1',[
      ('Init',[':all','ros',E,'norobot',E]),
      ('Exit',':close') ]),
    ('s1',[
      (':pair', ('rviz',['rviz',E]),
                ('kill',['C-c']) )  ]),
    ('s2',[
      ('ls',('ls',E)),
      ('nodes',['rostopic list',E]),
      ('topics',['rosnode list',E]) ]),
    ]
  exit_command= [E,'C-c']
  RunTerminalTab('Sample Launcher',terminals,exit_command)
