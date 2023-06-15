#!/usr/bin/python
#\file    read_discrete_in.py
#\brief   Test of read_discrete_input
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jun.14, 2023
#Use the server: $ python synchronous_server.py
from pymodbus.client.sync import ModbusTcpClient as ModbusClient

if __name__=='__main__':
  from _config import *

  #Connection to the server:
  client= ModbusClient(SERVER_URI, port=PORT)
  client.connect()

  print 'Read discrete input address 0, length 3'
  res_r= client.read_discrete_inputs(3, 3)
  print 'disc_in[0][:3]: {} (whole bits: {})'.format(res_r.bits[:3], res_r.bits)
  print '  response object:', res_r

  print 'Read discrete input address 3, length 9'
  res_r= client.read_discrete_inputs(3, 9)
  print 'disc_in[3][:9]: {} (whole bits: {})'.format(res_r.bits[:9], res_r.bits)
  print '  response object:', res_r

  #Disconnect from the server.
  client.close()

