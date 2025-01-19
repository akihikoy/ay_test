#!/usr/bin/python3

data_file= '/tmp/tmp_file_io.py.dat'
fp= open(data_file,'w')  #Do not use 'file'
for i in range(50):
  fp.write(str(i)+' '+str(i*i)+'\n')
fp.close()

i= 0
fp= open(data_file)  #Do not use 'file'
while True:
  line= fp.readline()
  if not line: break
  print('l.',i,line, end=' ')  #line terminates with new-line code
  i+=1

print(data_file,' closed? ',fp.closed)
fp.close()
print(data_file,' closed? ',fp.closed)

fp= open(data_file)  #Do not use 'file'
data= fp.read()
print('Data:',data.split())
fp.close()

#Delete the data file from /tmp
#import os
#os.remove(data_file)
