#!/usr/bin/python
#\file    terminal_tab8lib.py
#\brief   Simple Tab-Terminal GUI command launcher (library).
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Sep.10, 2017
#\version 0.2
#\date    Sep.3, 2018
#         Option (radio button) interface is added.
#\date    May.30, 2019
#         Option (combobox) interface is added.
#Requirements: tmux rxvt-unicode-256color

import sys, os
import signal
from PyQt4 import QtCore,QtGui,QtTest

class TTerminalTab(QtGui.QWidget):
  def __init__(self,title,widgets,exit_command,size=(800,400),horizontal=True,no_focus=True):
    QtGui.QWidget.__init__(self)
    self.pid= str(os.getpid())+'-'
    self.InitUI(title,widgets,exit_command,size,horizontal,no_focus)

  # Get a dict of option name: option content
  def ExpandOpt(self):
    opt= {name:rbgroup.checkedButton().text() for name,rbgroup in self.RBOptions.iteritems()}
    opt.update({name:cmbbx.currentText() for name,cmbbx in self.CBOptions.iteritems()})
    return opt

  def CmdToLambda(self,term,cmd):
    if cmd==':close':  return self.close
    if len(cmd)==0:  return lambda:None
    if cmd[0]==':all':  return lambda:self.SendCmdToAll([c.format(**self.ExpandOpt()) for c in cmd[1:]])
    return lambda:self.SendCmd(term,[c.format(**self.ExpandOpt()) for c in cmd])

  def InitUI(self,title,widgets,exit_command,size,horizontal,no_focus,grid_type='vhbox'):
    # Set window size.
    self.resize(*size)
    self.Processes= []
    self.TermProcesses= []

    # Set window title
    self.WinTitle= title
    self.setWindowTitle(self.WinTitle)

    self.ExitCommand= exit_command
    self.Widgets= widgets
    self.Terminals= [line for line in self.Widgets if isinstance(line[1],(tuple,list))]
    self.RBOptions= {}  #Option name: Option radio button group
    self.CBOptions= {}  #Option name: Option combobox
    self.term_to_idx= {term:r for r,(term,row) in enumerate(self.Terminals)}

    # Horizontal box layout
    if horizontal:  boxlayout= QtGui.QHBoxLayout()
    else:           boxlayout= QtGui.QVBoxLayout()
    self.setLayout(boxlayout)

    self.qttabs= self.MakeTabs()
    boxlayout.addWidget(self.qttabs)

    # Grid layout

    wg= QtGui.QWidget()
    grid= QtGui.QGridLayout()
    #if grid_type=='grid':  grid= QtGui.QGridLayout()
    #elif grid_type=='vhbox':  grid= QtGui.QVBoxLayout()
    wg.setLayout(grid)
    boxlayout.addWidget(wg)

    # Add widgets on grid
    for r,line in enumerate(self.Widgets):
      gcol= [0]
      if grid_type=='grid':
        def add_widget(w):
          grid.addWidget(w, r, gcol[0])
          gcol[0]+= 1
      elif grid_type=='vhbox':
        gline= QtGui.QHBoxLayout()
        def add_widget(w):
          if gcol[0]==0:
            grid.addWidget(w, r, 0)
            grid.addLayout(gline, r, 1)
            gcol[0]+= 1
          else:
            w.setSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
            #w.setMaximumWidth(5)
            #w.resize(w.sizeHint().width(), w.sizeHint().height()*2)
            gline.addWidget(w)
      if isinstance(line,(tuple,list)) and len(line)>1 and line[1]==':radio':
        name,_,options= line
        label= QtGui.QLabel()
        label.setText(name)
        label.setAlignment(QtCore.Qt.AlignCenter)
        add_widget(label)
        group= QtGui.QButtonGroup()
        for i,opt in enumerate(options):
          radbtn= QtGui.QRadioButton(opt)
          radbtn.setCheckable(True)
          if no_focus:  radbtn.setFocusPolicy(QtCore.Qt.NoFocus)
          if i==0:  radbtn.setChecked(True)
          group.addButton(radbtn,1)
          add_widget(radbtn)
        self.RBOptions[name]= group
      elif isinstance(line,(tuple,list)) and len(line)>1 and line[1]==':cmb':
        name,_,options= line
        label= QtGui.QLabel()
        label.setText(name)
        label.setAlignment(QtCore.Qt.AlignCenter)
        add_widget(label)
        cmbbx= QtGui.QComboBox(self)
        for opt in options:
          cmbbx.addItem(opt)
        cmbbx.setCurrentIndex(0)
        add_widget(cmbbx)
        self.CBOptions[name]= cmbbx
      elif isinstance(line,(tuple,list)) and len(line)>1 and isinstance(line[1],(tuple,list)):
        term,row= line
        btn0= QtGui.QPushButton('({term})'.format(term=term))
        btn0.setFlat(True)
        if no_focus:  btn0.setFocusPolicy(QtCore.Qt.NoFocus)
        btn0.clicked.connect(lambda clicked,term=term:self.ShowTermTab(term))
        add_widget(btn0)
        for commands in row:
          if commands[0]==':pair':
            name1,f1= commands[1][0],self.CmdToLambda(term,commands[1][1])
            name2,f2= commands[2][0],self.CmdToLambda(term,commands[2][1])
            btn= QtGui.QPushButton(name1)
            btn.setStyleSheet('padding:5px 10px 5px 10px')
            btn.setCheckable(True)
            if no_focus:  btn.setFocusPolicy(QtCore.Qt.NoFocus)
            btn.clicked.connect(lambda b,btn=btn,name1=name1,f1=f1,name2=name2,f2=f2:
                                  (f1(),btn.setText(name2)) if btn.isChecked() else (f2(),btn.setText(name1)))
            add_widget(btn)
          else:
            name,f= commands[0],self.CmdToLambda(term,commands[1])
            btn= QtGui.QPushButton(name)
            btn.setStyleSheet('padding:5px 10px 5px 10px')
            if no_focus:  btn.setFocusPolicy(QtCore.Qt.NoFocus)
            btn.clicked.connect(f)
            add_widget(btn)
      else:
        raise Exception('Unknown syntax:',line)
      if grid_type=='vhbox':
        gline.addSpacerItem(QtGui.QSpacerItem(1, 1, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding))

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
    return child

  def SendCmd(self, term, cmd):
    self.ShowTermTab(term)
    self.StartProc('tmux', ['send-keys', '-t', self.pid+term+':0'] + list(cmd))

  def SendCmdToAll(self, cmd):
    for r,(term,row) in enumerate(self.Terminals):
      self.ShowTermTab(term)
      self.StartProc('tmux', ['send-keys', '-t', self.pid+term+':0'] + list(cmd))

  def CreateTerminals(self):
    for r,(term,row) in enumerate(self.Terminals):
      self.qttabs.setCurrentIndex(r)
      self.TermProcesses.append(
        self.StartProc(
          'urxvt',
          ['-embed', str(self.qtterm[term].winId()),
            '-e', 'tmux', 'new', '-s', self.pid+term]) )
      #print 'new terminal proc:',self.pid+term,self.TermProcesses[-1].pid()
      #QtTest.QTest.qWait(100)
    QtTest.QTest.qWait(200)
    #QtTest.QTest.qWait(200*len(self.Terminals))
    #for r,(term,row) in enumerate(self.Terminals):
      #self.StartProc('tmux', ['send-keys', '-t', self.pid+term+':0'] + self.InitCommand)
    self.qttabs.setCurrentIndex(0)

  def Exit(self):
    for r,(term,row) in enumerate(self.Terminals):
      self.qttabs.setCurrentIndex(r)
      self.StartProc('tmux', ['send-keys', '-t', self.pid+term+':0'] + self.ExitCommand)
    QtTest.QTest.qWait(200)
    for r,(term,row) in enumerate(self.Terminals):
      self.qttabs.setCurrentIndex(r)
      self.StartProc('tmux', ['send-keys', '-t', self.pid+term+':0', 'exit', 'Enter'])
    QtTest.QTest.qWait(200)
    for proc in self.TermProcesses:
      os.kill(proc.pid(), signal.SIGTERM)

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

def RunTerminalTab(title,widgets,exit_command,size=(800,400),horizontal=True,no_focus=True):
  app= QtGui.QApplication(sys.argv)
  win= TTerminalTab(title,widgets,exit_command,size=size,horizontal=horizontal,no_focus=no_focus)
  signal.signal(signal.SIGINT, lambda signum,frame,win=win: (win.Exit(),QtGui.QApplication.quit()) )
  timer= QtCore.QTimer()
  timer.start(500)  # You may change this if you wish.
  timer.timeout.connect(lambda: None)  # Let the interpreter run each 500 ms.
  sys.exit(app.exec_())

if __name__=='__main__':
  E= 'Enter'
  widgets= [
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
  RunTerminalTab('Sample Launcher',widgets,exit_command)
