#!/usr/bin/python
#\file    terminal_tab8a.py
#\brief   Test of using the library terminal_tab7lib.py;
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Sep.02, 2018

from terminal_tab8lib import RunTerminalTab

if __name__=='__main__':
  E= 'Enter'
  widgets= [
    ('main1',[
      ('Init',[':all','ros',E,'norobot',E]),
      ('Exit',':close'), 
      (':pair', ('roscore',['roscore',E]),
                ('kill',['C-c']) )  ]),
    ('rviz',[
      (':pair', ('rviz',['rviz',E]),
                ('kill',['C-c']) )  ]),
    ('opt1',':cmb',['USB0','USB1']),
    ('s2',[
      ('ls',('ls',E)),
      ('ls opt1',('ls /dev/tty{opt1}',E)),
      ('ls opt2',('ls /dev/input/{opt2}',E)),
      ('nodes',['rostopic list',E]),
      ('topics',['rosnode list',E]) ]),
    ('opt2',':radio',['js0','js1']),
    ]
  exit_command= [E,'C-c']
  RunTerminalTab('Test Launcher',widgets,exit_command)
