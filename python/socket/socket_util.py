#!/usr/bin/python
#\file    socket_util.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Oct.24, 2016
#cf. http://stackoverflow.com/questions/17667903/python-socket-receive-large-amount-of-data
import struct

# Prefix each message with a 4-byte length (network byte order)
def send_msg(sock, msg):
  msg= struct.pack('>I', len(msg)+1) + msg + '\n'
  sock.sendall(msg)

# Read message length and unpack it into an integer
def recv_msg(sock):
  raw_msglen= recvall(sock, 4)
  if not raw_msglen:
    return None
  msglen= struct.unpack('>I', raw_msglen)[0]
  # Read the message data
  data= recvall(sock, msglen)
  return data[:-1] if data[-1]=="\n" else data

# Helper function to recv n bytes or return None if EOF is hit
def recvall(sock, n):
  data= ''
  while len(data) < n:
    packet= sock.recv(n - len(data))
    if not packet:
      return None
    data+= packet
  return data

