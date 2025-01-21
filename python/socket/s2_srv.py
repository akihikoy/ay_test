#!/usr/bin/python3
#\file    s2_srv.py
#\brief   Socket programming test: the server sends a dict data every 10 hz, and receives a string command to change the dict values.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jan.24, 2024

import socket
import json
import time
import select

data_dict= {"message": "Hello", "number": 1}

def server_loop(conn):
  global data_dict
  try:
    while True:
      # Send current data_dict to client at 10Hz
      data= (json.dumps(data_dict)+'\n').encode('utf-8')
      print('Sending {}...'.format(data))
      conn.sendall(data)
      time.sleep(0.1)  # 100ms for 10Hz

      # Check for incoming commands to change dict values
      ready = select.select([conn], [], [], 0.1)
      if ready[0]:
        data = conn.recv(1024).decode('utf-8')
        if data:
          print('Received: {}'.format(data))
          # Assuming command is in format "key:value"
          key, value = data.split(":")
          if key in data_dict and key == "number":
            data_dict[key] = int(value)
          else:
            data_dict[key] = value
  except Exception as e:
    print("Error: {}".format(e))
  finally:
    conn.close()

def main():
  HOST = 'localhost'
  PORT = 20001

  sock= socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  sock.bind((HOST, PORT))
  sock.listen(1)
  print('Waiting for a connection')
  connection, client_address= sock.accept()
  print('Connected by', client_address)
  server_loop(connection)

if __name__ == "__main__":
  main()
