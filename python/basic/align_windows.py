#!/usr/bin/python
#\file    align_windows.py
#\brief   Test code of aligning windows using wmctrl.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.14, 2023
from __future__ import print_function
import subprocess

def get_window_list():
  try:
    return subprocess.check_output(['wmctrl', '-l']).decode('utf-8').splitlines()
  except subprocess.CalledProcessError:
    return None

def get_window_id_by_title(window_list, title):
  for line in window_list:
    if title.lower() in line.lower():
      return line.split()[0]
  return None

positions= {
  'fvp_1_l-blob': '0,0,640,480',
  'fvp_1_r-blob': '648,0,640,480',
  'fvp_1_l-pxv': '0,545,640,480',
  'fvp_1_r-pxv': '648,545,640,480',
  'Robot Operation Panel': '1200,0,600,500',
  }

window_list= get_window_list()
for title, pos in positions.items():
  window_id= get_window_id_by_title(window_list, title)
  if window_id:
    subprocess.call(['wmctrl', '-i', '-r', window_id, '-e', '0,{}'.format(pos)])
  else:
    print('No window with title containing "{}" found.'.format(title))

print('Windows have been aligned.')
