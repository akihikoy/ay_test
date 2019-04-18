#!/usr/bin/python
import readline

histfile = ".pyhist"
try:
  readline.read_history_file(histfile)
except IOError:
  pass

readline.parse_and_bind('tab: complete')
while True:
  line = raw_input('"stop" to quit > ')
  if line == 'stop':
    break
  print '  entered: ',line

readline.write_history_file(histfile)