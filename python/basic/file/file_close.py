#!/usr/bin/python
import weakref
fp_ref= None
data_file= '/tmp/tmp_file_io.py.dat'

def Func1():
  fp= open(data_file,'w')

  global fp_ref
  fp_ref= weakref.ref(fp)
  #fp_ref= lambda:fp
  print data_file,' closed? ',fp_ref().closed

  for i in range(50):
    fp.write(str(i)+' '+str(i*i)+'\n')
  #fp.close()

Func1()
print data_file,' closed? ',fp_ref().closed
