#!/usr/bin/python3
#\file    test_sched_setscheduler.py
#\brief   Testing sched_setscheduler system call in Python.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jun.25, 2018

'''
refs.
https://gist.github.com/syohex/8101d069fff5a3793155
http://stts.hatenablog.com/entry/20070807/1186497164

Need to be executed as a superuser.
'''

import ctypes
import os

SCHED_OTHER= ctypes.c_int(0)
SCHED_FIFO= ctypes.c_int(1)
SCHED_RR= ctypes.c_int(2)

libc= ctypes.CDLL("libc.so.6")
print("Before my scheduler=", libc.sched_getscheduler(os.getpid()))

param= ctypes.c_int(99)
err= libc.sched_setscheduler(os.getpid(), SCHED_FIFO, ctypes.byref(param))
if err != 0:
  print("errno=", ctypes.get_errno())

print("After my scheduler=", libc.sched_getscheduler(os.getpid()))

