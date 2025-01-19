#!/usr/bin/python3
#\file    run_command.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jun.10, 2015
import subprocess, os

if __name__=='__main__':
  print('Run-1')
  os.system('''qplot -x 'sin(x)' ''')
  print('Done-1')

  print('Run-2')
  #subprocess.call('''qplot -x 'sin(x)' ''')  #No such file
  subprocess.call('''qplot -x 'sin(x)' ''', shell=True)
  print('Done-2')

  print('Run-3')
  subprocess.call(['qplot','-x', '''sin(x)'''])
  print('Done-3')

  print('Run-4')
  os.system('''qplot -x2 'aaa' 'sin(x)' &''')
  print('Done-4')

  print('Run-5')
  subprocess.call('''qplot -x2 'aaa' 'sin(x)' &''', shell=True)
  print('Done-5')


