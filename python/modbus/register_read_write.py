#!/usr/bin/python3
#\file    register_read_write.py
#\brief   Read from holding register, write to register.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jun.14, 2023
#Use one of the servers:
#  $ ./synchronous_server.py
#  $ ./sync_server_2.py
from pymodbus.client.sync import ModbusTcpClient as ModbusClient
from pymodbus.pdu import ExceptionResponse
import sys

if __name__=='__main__':
  from _config import *

  #Connection to the server:
  client= ModbusClient(SERVER_URI, port=PORT)
  client.connect()

  '''
  address can be [0,99-count] (otherwise the response is exception/IllegalAddress).
  This may be due to the server configuration (the block size is 100 (0-99)-->99-count).
  value can be [0, 65535], integer (floating number is casted to int)
  '''
  address= 1
  count= 1
  print('Read holding registers address {}, count {}'.format(address, count))
  res_r= client.read_holding_registers(address, count)
  if not isinstance(res_r, ExceptionResponse):
    print('holding_registers[{}][:{}]: {}'.format(address, count, res_r.registers))
  print('  response object:', res_r)

  value= 10
  print('Write register address {}, value {}'.format(address, value))
  res_w= client.write_register(address, value)
  print('  response object:', res_w)

  print('Read holding registers address {}, count {}'.format(address, count))
  res_r= client.read_holding_registers(address, count)
  if not isinstance(res_r, ExceptionResponse):
    print('holding_registers[{}][:{}]: {}'.format(address, count, res_r.registers))
  print('  response object:', res_r)

  value= 1024
  print('Write register address {}, value {}'.format(address, value))
  res_w= client.write_register(address, value)
  print('  response object:', res_w)

  print('Read holding registers address {}, count {}'.format(address, count))
  res_r= client.read_holding_registers(address, count)
  if not isinstance(res_r, ExceptionResponse):
    print('holding_registers[{}][:{}]: {}'.format(address, count, res_r.registers))
  print('  response object:', res_r)

  print('-------------------')

  '''
  address can be [0,99-count] (otherwise the response is exception/IllegalAddress).
  This may be due to the server configuration (the block size is 100 (0-99)-->99-count).
  value can be [0, 65535], integer
  '''
  address= 2
  count= 10
  print('Read holding registers address {}, count {}'.format(address, count))
  res_r= client.read_holding_registers(address, count)
  if not isinstance(res_r, ExceptionResponse):
    print('holding_registers[{}][:{}]: {}'.format(address, count, res_r.registers))
  print('  response object:', res_r)

  #WARNING: Works with Python2, Python3: struct.error: required argument is not an integer.
  #NOTE that in Python2, the floating values are converted to integers.
  if sys.version.startswith('2'):
    value= [i**5.047 for i in range(10)]
    print('Write registers address {}, value {}'.format(address, value))
    res_w= client.write_registers(address, value)
    print('  response object:', res_w)

    print('Read holding registers address {}, count {}'.format(address, count))
    res_r= client.read_holding_registers(address, count)
    if not isinstance(res_r, ExceptionResponse):
      print('holding_registers[{}][:{}]: {}'.format(address, count, res_r.registers))
    print('  response object:', res_r)
    print('-------------------')

  value= [int(i**5.047) for i in range(10)]
  print('Write registers address {}, value {}'.format(address, value))
  res_w= client.write_registers(address, value)
  print('  response object:', res_w)

  print('Read holding registers address {}, count {}'.format(address, count))
  res_r= client.read_holding_registers(address, count)
  if not isinstance(res_r, ExceptionResponse):
    print('holding_registers[{}][:{}]: {}'.format(address, count, res_r.registers))
  print('  response object:', res_r)

  #Disconnect from the server.
  client.close()
