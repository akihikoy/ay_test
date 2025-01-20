#!/usr/bin/python3
#\file    srv3_client.py
#\brief   Client using sync_server_3.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jun.24, 2023
#Use one of the servers:
#  $ ./sync_server_3.py
#  $ ./sync_server_3a.py
from pymodbus.client.sync import ModbusTcpClient as ModbusClient
from pymodbus.pdu import ExceptionResponse
from kbhit2 import KBHAskGen
from sync_server_3 import *

if __name__=='__main__':
  from _config import *

  #Connection to the server:
  client= ModbusClient(SERVER_URI, port=PORT)
  client.connect()

  address= 0
  count= 10
  while True:
    res_r= client.read_input_registers(address, count)
    if isinstance(res_r, ExceptionResponse):
      print('Server trouble.')
    print('Values:', res_r.registers)
    print('Type action:')
    key= KBHAskGen('q','0','1','2','3','4')
    if key=='q':  break
    res_w= client.write_register(ADDR_ACTION_REQ-1, int(key))

  #Disconnect from the server.
  client.close()
