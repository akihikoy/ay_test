#!/usr/bin/python3
#\file    s1_srv.py
#\brief   Test of socket communication (server)
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

if __name__=='__main__':

  # Create a TCP/IP socket
  sock= socket.socket(socket.AF_INET, socket.SOCK_STREAM)

  # Bind the socket to the port
  server_address= ('localhost', 20000)
  print('Starting server %s:%s' % server_address)
  sock.bind(server_address)

  # Listen for incoming connections
  sock.listen(1)

  # Wait for a connection
  print('Waiting for a connection')
  connection, client_address= sock.accept()

  try:
    print('Got a client: %s:%s' % client_address)

    # Receive the data in small chunks and retransmit it
    while True:
      #data= connection.recv(16)
      data= socket_util.recv_msg(connection).decode('utf-8')
      print('Received: "{}"'.format(data))
      if data:
        new_data= 'Your data is "{}"'.format(data)
        #connection.sendall(new_data)
        socket_util.send_msg(connection, new_data.encode('utf-8'))
      else:
        print('No more data from', client_address)
        break

  finally:
    # Clean up the connection
    print('Closing connection')
    connection.close()

