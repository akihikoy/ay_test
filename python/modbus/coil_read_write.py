#!/usr/bin/python
#\file    coil_read_write.py
#\brief   Write and read coil through Modbus.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jun.14, 2023
from pymodbus.client.sync import ModbusTcpClient as ModbusClient

if __name__=='__main__':
  from _config import *

  #Connection to the server:
  client= ModbusClient(SERVER_URI, port=PORT)
  client.connect()

  print 'Read coil address 1, length 1'
  res_r= client.read_coils(1, 1)
  print 'coil[1][1]: {} (whole bits: {})'.format(res_r.bits[0], res_r.bits)
  print '  response object:', res_r

  print 'Write coil address 1, value True'
  res_w= client.write_coil(1, True)
  print '  response object:', res_w

  print 'Read coil address 1, length 1'
  res_r= client.read_coils(1, 1)
  print 'coil[1][1]: {} (whole bits: {})'.format(res_r.bits[0], res_r.bits)
  print '  response object:', res_r

  print 'Write coil address 1, value False'
  res_w= client.write_coil(1, False)
  print '  response object:', res_w

  print 'Read coil address 1, length 1'
  res_r= client.read_coils(1, 1)
  print 'coil[1][1]: {} (whole bits: {})'.format(res_r.bits[0], res_r.bits)
  print '  response object:', res_r

  print '-------------------'

  print 'Read coil address 3, length 9'
  res_r= client.read_coils(3, 9)
  print 'coil[3][:9]: {} (whole bits: {})'.format(res_r.bits[:9], res_r.bits)
  print '  response object:', res_r

  print 'Write coil address 3, value [True]*9'
  res_w= client.write_coils(3, [True]*9)
  print '  response object:', res_w

  print 'Read coil address 3, length 9'
  res_r= client.read_coils(3, 9)
  print 'coil[3][:9]: {} (whole bits: {})'.format(res_r.bits[:9], res_r.bits)
  print '  response object:', res_r

  print 'Write coil address 3, value [True,False]*4+[True]'
  res_w= client.write_coils(3, [True,False]*4+[True])
  print '  response object:', res_w

  print 'Read coil address 3, length 9'
  res_r= client.read_coils(3, 9)
  print 'coil[3][:9]: {} (whole bits: {})'.format(res_r.bits[:9], res_r.bits)
  print '  response object:', res_r

  #Disconnect from the server.
  client.close()
