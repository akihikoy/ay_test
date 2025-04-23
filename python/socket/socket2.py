#!/usr/bin/python3
#\file    socket2.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Mar.14, 2025

import socket
import struct

HOST = '0.0.0.0'
PORT = 8000

FMT = '>IIHHHHHHI'
DELIM = b'\r\n'

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
  print('connecting...')
  s.bind((HOST, PORT))
  s.listen()
  conn,addr= s.accept()
  print('connected')
  is_regular = False
  with conn:
    while True:
      if not is_regular:
        while True:
          data = conn.recv(2)
          if not data:
            break
          if data==DELIM:
            break
      data = conn.recv(26)
      if not data:
        break
      if len(data)!=26:
        print(f'incomplete data (len={len(data)})')
        continue
      if data[-2:]!=DELIM:
        print('invalid delimiter')
        continue
      print(f'data len:{len(data)}')
      values = struct.unpack(FMT, data[:24])
      print('received:',values)
