#!/usr/bin/python3
#\file    motoman_client.py
#\brief   Client test for the Modbus server on the Motoman robot.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Sep.20, 2023

from pymodbus.client.sync import ModbusTcpClient as ModbusClient
from pymodbus.pdu import ExceptionResponse
from kbhit2 import KBHAskGen

#SERVER_URI= '192.168.250.81'
#SERVER_URI= '10.10.6.204'
SERVER_URI= '192.168.1.100'  #CRX
PORT= 502

if __name__=='__main__':
  #Connection to the server:
  client= ModbusClient(SERVER_URI, port=PORT)
  client.connect()

  print('Connection established: {}'.format(client))

  try:
    address= 4  #Should be within the M-Register Address setting of Motoman.
    count= 10  #Should be the same or smaller than the M-Register Size (word) setting of Motoman.

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
      value= input('Type value:')
      if value=='q':  break
      res_w= client.write_register(address, int(value))
      print('Write: {}'.format(res_w))

  finally:
    #Disconnect from the server.*
    client.close()
