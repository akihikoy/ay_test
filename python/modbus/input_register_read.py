#!/usr/bin/python
#\file    input_register_read.py
#\brief   Test of read_input_register.
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

  print 'Read input registers address 2, length 10'
  res_r= client.read_input_registers(2, 10)
  print 'input_registers[2][:10]: {}'.format(res_r.registers)
  print '  response object:', res_r

  #Disconnect from the server.
  client.close()
