#!/usr/bin/python
#\file    register_read_write.py
#\brief   Read from holding register, write to register.
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

  print 'Read holding registers address 1, length 1'
  res_r= client.read_holding_registers(1, 1)
  print 'holding_registers[1][1]: {}'.format(res_r.registers)
  print '  response object:', res_r

  print 'Write register address 1, value 10'
  res_w= client.write_register(1, 10)
  print '  response object:', res_w

  print 'Read holding registers address 1, length 1'
  res_r= client.read_holding_registers(1, 1)
  print 'holding_registers[1][1]: {}'.format(res_r.registers)
  print '  response object:', res_r

  print 'Write register address 1, value 1024'
  res_w= client.write_register(1, 1024)
  print '  response object:', res_w

  print 'Read holding registers address 1, length 1'
  res_r= client.read_holding_registers(1, 1)
  print 'holding_registers[1][1]: {}'.format(res_r.registers)
  print '  response object:', res_r

  print '-------------------'

  print 'Read holding registers address 2, length 10'
  res_r= client.read_holding_registers(2, 10)
  print 'holding_registers[2][:10]: {}'.format(res_r.registers)
  print '  response object:', res_r

  print 'Write registers address 2, value {}'.format([i**2 for i in range(10)])
  res_w= client.write_registers(2, [i**2 for i in range(10)])
  print '  response object:', res_w

  print 'Read holding registers address 2, length 10'
  res_r= client.read_holding_registers(2, 10)
  print 'holding_registers[2][:10]: {}'.format(res_r.registers)
  print '  response object:', res_r

  #Disconnect from the server.
  client.close()
