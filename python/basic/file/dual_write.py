#!/usr/bin/python
#\file    dual_write.py
#\brief   Virtual file pointer to write the same contents into a file and std-out.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.20, 2019
import sys

'''
Virtual file pointer to write the same contents into a file A and std-out.
Basically it behaves as if a file object of A opened as 'w' mode.
Usage:
  with DualWriter('file_path') as fp:
    fp.write('hello DualWriter;\n')
'''
class TDualWriter(file):
  def __init__(self,filename):
    super(TDualWriter,self).__init__(filename,'w')
  def flush(self):
    sys.stdout.flush()
    super(TDualWriter,self).flush()
  def write(self,str):
    sys.stdout.write(str)
    super(TDualWriter,self).write(str)
  def writelines(self,sequence):
    sys.stdout.writelines(sequence)
    super(TDualWriter,self).writelines(sequence)
def DualWriter(filename):
  return TDualWriter(filename)


if __name__=='__main__':
  with DualWriter('/tmp/hoge.dat') as fp:
    print 'status:',fp.closed
    fp.write('hello DualWriter;\n')
    fp.write('test test test...;\n')

  print 'status:',fp.closed
