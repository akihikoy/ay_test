#!/usr/bin/python
#\file    read_discrete_in.py
#\brief   Test of read_discrete_input
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jun.14, 2023
#Use the server: $ python synchronous_server.py
from pymodbus.client.sync import ModbusTcpClient as ModbusClient
from pymodbus.pdu import ExceptionResponse

if __name__=='__main__':
  from _config import *

  #Connection to the server:
  client= ModbusClient(SERVER_URI, port=PORT)
  client.connect()

  '''
  address can be [0,99-count] (otherwise the response is exception/IllegalAddress).
  This may be due to the server configuration (the block size is 100 (0-99)-->99-count).
  '''
  address= 0
  count= 3
  print 'Read discrete input address {}, count {}'.format(address, count)
  res_r= client.read_discrete_inputs(address, count)
  if not isinstance(res_r, ExceptionResponse):
    print 'disc_in[{}][:{}]: {} (whole bits: {})'.format(address, count, res_r.bits[:count], res_r.bits)
  print '  response object:', res_r

  address= 3
  count= 9
  print 'Read discrete input address {}, count {}'.format(address, count)
  res_r= client.read_discrete_inputs(address, count)
  if not isinstance(res_r, ExceptionResponse):
    print 'disc_in[{}][:{}]: {} (whole bits: {})'.format(address, count, res_r.bits[:count], res_r.bits)
  print '  response object:', res_r

  #Disconnect from the server.
  client.close()

