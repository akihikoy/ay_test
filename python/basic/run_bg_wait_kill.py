#!/usr/bin/python3
#\file    run_bg_wait_kill.py
#\brief   Run an external program as a subprocess in background, wait, and kill.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Apr.19, 2021
import subprocess
import time
import sys

if __name__=='__main__':
  p= subprocess.Popen(['roscore'])
  #p= subprocess.Popen(['roslaunch', 'ay_util', 'ur_selector.launch', 'robot_code:=UR3DxlpO2_Fork1_SIM', 'jsdev:=/dev/input/js0', 'dxldev:=/dev/ttyUSB1'])

  #exit_code = p.wait()

  print('waiting...', end=' ')
  for i in range(10):
    print(i, end=' ')
    sys.stdout.flush()
    time.sleep(0.5)
  print('done')

  #p.wait()
  p.terminate()
  #p.kill()  #NOTE: Not good for terminating roscore as it does not kill rosmaster.

  #A way to wait for the process termination.
  #p.wait()
  #print 'Process terminated'

  #Another way to wait for the process termination.
  while p.poll() is None:
    print('Process still running...')
    time.sleep(0.1)
  print('Process terminated')

