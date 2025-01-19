#!/usr/bin/python3
import sys

if len(sys.argv)==1:
  sys.exit(1)

inpre=False
fp= file(sys.argv[1])
while True:
  line= fp.readline()
  if not line: break
  if line[:3]=='>||':
    inpre= True
    print('')
  elif inpre and line[:2]=='}}':
    inpre= False
    print('')
  elif line[:6]=='<body>':
    print('%s\n\n\n'%line)
  elif line[:7]=='</body>':
    print('\n\n%s\n'%line)
  else:
    if inpre:
      print(' %s'%line, end=' ')
    else:
      print('%s'%line, end=' ')
