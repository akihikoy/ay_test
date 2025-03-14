#!/usr/bin/python3
#\file    socket1.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Mar.14, 2025

import socket
import struct

HOST = '0.0.0.0'
PORT = 8000

FMT = '<IIHHHHHHI'

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
  print('connecting...')
  s.bind((HOST, PORT))
  s.listen()
  conn,addr= s.accept()
  print('connected')
  with conn:
    while True:
      data = conn.recv(26)
      if not data:
        break
      if len(data)!=24:
        print(f'incomplete data (len={len(data)})')
        continue
      print(f'data len:{len(data)}')
      values = struct.unpack(FMT, data)
      print('received:',values)

