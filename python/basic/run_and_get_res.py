#!/usr/bin/python3
#\file    run_and_get_res.py
#\brief   Run command and get the result as a string.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jun.05, 2020
import subprocess

if __name__=='__main__':
  p= subprocess.Popen(['ls', '-rtl', '.'], stdout=subprocess.PIPE)
  (stdout, stderr)= map(lambda b:b.decode('utf-8') if b is not None else None, p.communicate())
  exit_code = p.wait()

  print('Got result:')
  print('  stdout:',stdout.strip().split('\n')[-3:])
  print('  stderr:',stderr)

  p= subprocess.Popen(['rospack', 'find', 'fingervision'], stdout=subprocess.PIPE)
  (stdout, stderr)= map(lambda b:b.decode('utf-8') if b is not None else None, p.communicate())
  exit_code = p.wait()

  print('Got result:')
  print('  stdout:',stdout.strip())
  print('  stderr:',stderr)

  #Shorter form:
  def ExtCmd(cmd):
    p= subprocess.Popen(cmd, stdout=subprocess.PIPE)
    (stdout, stderr)= map(lambda b:b.decode('utf-8') if b is not None else None, p.communicate())
    exit_code= p.wait()
    return stdout.strip()

  print('Got result:')
  print('  stdout:',ExtCmd(['rospack', 'find', 'fingervision']))
