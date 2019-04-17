#!/usr/bin/python
import math

def EstStrConvert(v_str):
  try:
    return int(v_str)
  except ValueError:
    pass
  try:
    return float(v_str)
  except ValueError:
    pass
  if v_str=='True' or v_str=='true' :  return True
  if v_str=='False' or v_str=='false':  return False
  try:
    x=[]
    for v in v_str.split(' '):
      x.append(float(v))
    return x
  except ValueError:
    pass
  try:
    x=[]
    for v in v_str.split(','):
      x.append(float(v))
    return x
  except ValueError:
    pass
  try:
    x=[]
    for v in v_str.split('\t'):
      x.append(float(v))
    return x
  except ValueError:
    pass
  return v_str


while True:
  s= raw_input('q or any-string > ')
  value= EstStrConvert(s)
  print '  type: ',type(value)
  print '  value: ',value
  if s=='q':
    break

