#!/usr/bin/python3
#\file    terminal_tab5a.py
#\brief   Test of using the library terminal_tab5lib.py;
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Sep.10, 2017

from terminal_tab5lib import RunTerminalTab

if __name__=='__main__':
  E= 'Enter'
  terminals= [
    ('main1',[
      ('Init',[':all','ros',E,'norobot',E]),
      ('Exit',':close') ]),
    ('rviz',[
      (':pair', ('rviz',['rviz',E]),
                ('kill',['C-c']) )  ]),
    ('s2',[
      ('ls',('ls',E)),
      ('nodes',['rostopic list',E]),
      ('topics',['rosnode list',E]) ]),
    ]
  exit_command= [E,'C-c']
  RunTerminalTab('Test Launcher',terminals,exit_command)
