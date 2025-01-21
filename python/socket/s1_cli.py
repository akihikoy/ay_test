#!/usr/bin/python3
#\file    s1_cli.py
#\brief   Test of socket communication (client)
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Oct.24, 2016
#cf. https://pymotw.com/2/socket/tcp.html
#cf. http://stackoverflow.com/questions/17667903/python-socket-receive-large-amount-of-data

# Also designed to communicate with Prolog.
# Simple protocol to communicate with Prolog:
# Message format = [Len|Body|Term].
#   Len: big-endian 4 byte integer that shows length of [Body|Term] (len(Body)+1).
#   Body: string that should not include Term="\n".
#   Term: terminal code "\n" (length=1).
# cf. ../../prolog/socket2_*.pl

import socket
import socket_util
import readline

if __name__=='__main__':

  # Create a TCP/IP socket
  sock= socket.socket(socket.AF_INET, socket.SOCK_STREAM)

  # Connect the socket to the port where the server is listening
  server_address= ('localhost', 20000)
  print('Connecting to %s:%s' % server_address)
  sock.connect(server_address)

  try:
    while True:
      # Send data
      message= input('quit or msg > ')
      if message=='quit':  break

      print('Sending: "%s"' % message)
      #sock.sendall(message)
      socket_util.send_msg(sock, message.encode('utf-8'))

      ## Look for the response
      #amount_received = 0
      #amount_expected = len(message)

      #while amount_received < amount_expected:
        #data = sock.recv(16)
        #amount_received += len(data)
        #print 'received "%s"' % data

      data= socket_util.recv_msg(sock).decode('utf-8')
      print('Received: "{}"'.format(data))

  finally:
    print('Closing connection')
    sock.close()


