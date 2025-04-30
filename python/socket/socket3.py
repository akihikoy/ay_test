#!/usr/bin/python3
#\file    socket3.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Mar.14, 2025

import socket
import time
import struct

HOST= '10.10.6.210'
PORT= 8700

# <: little-endian
# i: signed 32bit
# I: unsigned 32bit
# H: unsigned 16bit
FMT= '<iiiIHHH'
DELIM= b'\r\n'

while True:
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    try:
      print(f"Connecting to {HOST}:{PORT}...")
      s.connect((HOST, PORT))
      print("Connection established.")

      while True:
        data = s.recv(1024)
        if not data:
          print("Connection closed by PLC.")
          break

        if len(data)==22 and data[-2:]==DELIM:
          values= struct.unpack(FMT, data)
          values_d= dict(inspection_count=values[0],
                         current_position=values[1],
                         load_value=values[2],
                         elapsed_time_s=3600*values[3]+60*values[4]+values[5])
          print(f"Received: {values}")
          print(f"Received: {values_d}")
          print(f"Received: {data}")

    except (ConnectionRefusedError, ConnectionResetError, OSError) as e:
      print(f'Connection error ({e}), retrying...')
      #time.sleep(2)
    except KeyboardInterrupt:
      print("User terminated the program.")
      break

  print("Reattempting connection...")
  time.sleep(2)



