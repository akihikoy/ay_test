#!/usr/bin/python
#\file    motoman_client.py
#\brief   Client test for the Modbus server on the Motoman robot.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Sep.20, 2023
from __future__ import print_function
from pymodbus.client.sync import ModbusTcpClient as ModbusClient
from pymodbus.pdu import ExceptionResponse
from kbhit2 import KBHAskGen

SERVER_URI= '192.168.250.81'
PORT= 502

if __name__=='__main__':
  #Connection to the server:
  client= ModbusClient(SERVER_URI, port=PORT)
  client.connect()

  print('Connection established: {}'.format(client))

  try:
    address= 0
    count= 10

    while True:
      #Read registers:
      #res_r= client.read_input_registers(address, count)
      res_r= client.read_holding_registers(address, count)
      print('Read: {}'.format(res_r))
      if isinstance(res_r, ExceptionResponse):
        print('--Server trouble.')
      else:
        print('--Values: {}'.format(res_r.registers))

      #Write registers:
      value= raw_input('Type value:')
      if value=='q':  break
      res_w= client.write_register(address, int(value))
      print('Write: {}'.format(res_w))

  finally:
    #Disconnect from the server.*
    client.close()
