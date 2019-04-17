#!/usr/bin/python
import sys,re

if len(sys.argv)==1:
  sys.exit(1)

rdate= re.compile(r'date\s*\=\s*\"([0-9\-]+)\"')
rexp= re.compile(r'^\*[0-9]+\*((\[[^\[\]]*\])*)\s*(.*)$')
#rexp= re.compile(r'^\*[0-9]*\*')

date=''
fp= file(sys.argv[1])
while True:
  line= fp.readline()
  if not line: break

  md= rdate.search(line)
  if md:
    date= md.group(1)

  m= rexp.search(line)
  if m:
    #print m.group(0)
    #print 'Tags:',m.group(1),'Title:',m.group(3)
    #print m.groups()
    print """\
article/%s

#lang(en)
#title(%s)
#lang(ja)
#title(%s)
#lang()
#tags(%s)
RIGHT:%s
""" % (m.group(3),m.group(3),m.group(3),m.group(1),date)
  else:
    print line,
