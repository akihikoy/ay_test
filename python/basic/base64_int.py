#!/usr/bin/python3
#\file    base64_int.py
#\brief   Encoding a sequence of numbers with base64.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.25, 2021
import base64

if __name__=='__main__':
  def PrintEq(x):  print('%s= %r' % (x, eval(x)))
  s= '1629876842981456041'
  s_i= int(s)
  s_bytes= s_i.to_bytes((s_i.bit_length()+7)//8, 'big')
  s_b64= base64.b64encode(s_bytes)
  s_decoded= int.from_bytes(base64.b64decode(s_b64), 'big')

  PrintEq('s')
  PrintEq('len(s)')
  PrintEq('s_i')
  PrintEq('s_bytes')
  PrintEq('s_b64')
  PrintEq('len(s_b64)')
  PrintEq('s_decoded')
