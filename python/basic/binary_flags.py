#!/usr/bin/python
#\file    binary_flags.py
#\brief   Test of binary flags.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.15, 2023
from __future__ import print_function

if __name__=='__main__':

  #List of available flags. NOTE: The value should be a power of 2.
  FLAG_LIST={
    'FLAG_A': 1,
    'FLAG_B': 2,
    'FLAG_C': 4,
    'FLAG_D': 8,
    }

  def p(flags):
    print('flags= {} ({})'.format(bin(flags),flags))
    print('  details: {}'.format({key:bool(flags&f) for key,f in FLAG_LIST.iteritems()}))

  flags= 0b00
  p(flags)
  flags|= FLAG_LIST['FLAG_A']
  p(flags)
  flags|= FLAG_LIST['FLAG_C'] | FLAG_LIST['FLAG_D']
  p(flags)
  flags= FLAG_LIST['FLAG_C'] | FLAG_LIST['FLAG_B']
  p(flags)

