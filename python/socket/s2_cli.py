#!/usr/bin/python
#\file    s2_cli.py
#\brief   Socket programming test: the server sends a dict data every 10 hz, and receives a string command to change the dict values.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jan.24, 2024
import socket
import json
import threading

def receive_data(sock, state):
  """Background thread function for receiving data continuously."""
  try:
    while state['running']:
      data = sock.recv(1024)
      if data:
        print 'Received', data.decode('utf-8')
      else:
        break  # Connection closed
  except Exception as e:
     print "Error in receive_data: {}".format(e)
  #finally:
    #sock.close()

def main():
  HOST = 'localhost'
  PORT = 20001

  sock= socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  print 'Connecting to %s:%s' % (HOST, PORT)
  try:
    sock.connect((HOST, PORT))
    state= {'running':True}
    th= threading.Thread(target=receive_data, args=(sock,state))
    th.start()
    while True:
      #data = sock.recv(1024)
      #print 'Received', data.decode('utf-8')
      # Example command to change the dict values
      command = raw_input("Enter command (key:value) or exit: ")
      print 'command:', command
      if command == "exit":
        break
      sock.sendall(command.encode('utf-8'))
  finally:
    state['running']= False
    th.join()
    print 'Closing connection'
    sock.close()

if __name__ == "__main__":
  main()
