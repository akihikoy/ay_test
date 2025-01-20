#!/usr/bin/python3
#\file    input_register_read.py
#\brief   Test of read_input_register.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jun.14, 2023
#Use one of the servers:
#  $ ./synchronous_server.py
#  $ ./sync_server_2.py
from pymodbus.client.sync import ModbusTcpClient as ModbusClient
from pymodbus.pdu import ExceptionResponse

if __name__=='__main__':
  from _config import *

  #Connection to the server:
  client= ModbusClient(SERVER_URI, port=PORT)
  client.connect()

  '''
  address can be [0,89] (otherwise the response is exception/IllegalAddress).
  This may be due to the server configuration (the block size is 100 (0-99)-->99-count=89).
  '''
  address= 2
  count= 10
  print('Read input registers address {}, count {}'.format(address, count))
  res_r= client.read_input_registers(address, count)
  if not isinstance(res_r, ExceptionResponse):
    print('input_registers[{}][:{}]: {}'.format(address, count, res_r.registers))
  print('  response object:', res_r)

  #Disconnect from the server.
  client.close()
