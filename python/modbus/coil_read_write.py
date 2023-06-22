#!/usr/bin/python
#\file    coil_read_write.py
#\brief   Write and read coil through Modbus.
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
  print 'Connected to a Modbus server: {}, client: {}'.format(SERVER_URI, client)

  '''
  address can be [0,98] (otherwise the response is exception/IllegalAddress).
  This may be due to the server configuration (the block size is 100 (0-99)-->99-count=98).
  '''
  address= 1
  count= 1
  print 'Read coil address {}, count {}'.format(address, count)
  res_r= client.read_coils(address, count)
  if not isinstance(res_r, ExceptionResponse):
    print 'coil[{}][:{}]: {} (whole bits: {})'.format(address, count, res_r.bits[:count], res_r.bits)
  print '  response object:', res_r

  value= True
  print 'Write coil address {}, value {}'.format(address, value)
  res_w= client.write_coil(address, value)
  print '  response object:', res_w

  print 'Read coil address {}, count {}'.format(address, count)
  res_r= client.read_coils(address, count)
  if not isinstance(res_r, ExceptionResponse):
    print 'coil[{}][:{}]: {} (whole bits: {})'.format(address, count, res_r.bits[:count], res_r.bits)
  print '  response object:', res_r

  value= False
  print 'Write coil address {}, value {}'.format(address, value)
  res_w= client.write_coil(address, value)
  print '  response object:', res_w

  print 'Read coil address {}, count {}'.format(address, count)
  res_r= client.read_coils(address, count)
  if not isinstance(res_r, ExceptionResponse):
    print 'coil[{}][:{}]: {} (whole bits: {})'.format(address, count, res_r.bits[:count], res_r.bits)
  print '  response object:', res_r

  print '-------------------'

  '''
  address can be [0,90] (otherwise the response is exception/IllegalAddress).
  This may be due to the server configuration (the block size is 100 (0-99)-->99-count=90).
  '''
  address= 3
  count= 9
  print 'Read coil address {}, count {}'.format(address, count)
  res_r= client.read_coils(address, count)
  if not isinstance(res_r, ExceptionResponse):
    print 'coil[{}][:{}]: {} (whole bits: {})'.format(address, count, res_r.bits[:count], res_r.bits)
  print '  response object:', res_r

  value= [True]*9
  print 'Write coil address {}, value {}'.format(address, value)
  res_w= client.write_coils(address, value)
  print '  response object:', res_w

  print 'Read coil address {}, count {}'.format(address, count)
  res_r= client.read_coils(address, count)
  if not isinstance(res_r, ExceptionResponse):
    print 'coil[{}][:{}]: {} (whole bits: {})'.format(address, count, res_r.bits[:count], res_r.bits)
  print '  response object:', res_r

  value= [True,False]*4+[True]
  print 'Write coil address {}, value {}'.format(address, value)
  res_w= client.write_coils(address, value)
  print '  response object:', res_w

  print 'Read coil address {}, count {}'.format(address, count)
  res_r= client.read_coils(address, count)
  if not isinstance(res_r, ExceptionResponse):
    print 'coil[{}][:{}]: {} (whole bits: {})'.format(address, count, res_r.bits[:count], res_r.bits)
  print '  response object:', res_r

  #Disconnect from the server.
  client.close()
