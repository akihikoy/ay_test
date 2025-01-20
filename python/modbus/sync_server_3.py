#!/usr/bin/python3
#\file    sync_server_3.py
#\brief   Synchronous Modbus TCP server test 3.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jun.24, 2023
import threading
import copy
import sys
from kbhit2 import TKBHit
from rate_adjust import TRate
from pymodbus.server.sync import StartTcpServer
from pymodbus.device import ModbusDeviceIdentification
from pymodbus.datastore import ModbusSequentialDataBlock, ModbusSparseDataBlock
from pymodbus.datastore import ModbusSlaveContext, ModbusServerContext

def MergeDict(d_base, d_new, allow_new_key=True):
  if isinstance(d_new, (list,tuple)):
    for d_new_i in d_new:
      MergeDict(d_base, d_new_i)
  else:
    for k_new,v_new in d_new.items():
      if not allow_new_key and k_new not in d_base:
        raise Exception('MergeDict: Unexpected key:',k_new)
      if k_new in d_base and (isinstance(v_new,dict) and isinstance(d_base[k_new],dict)):
        MergeDict(d_base[k_new], v_new)
      else:
        d_base[k_new]= v_new
  return d_base  #NOTE: d_base is overwritten. Returning it is for the convenience.

#ref. ModbusSequentialDataBlock:
#https://github.com/pymodbus-dev/pymodbus/blob/2be46d/pymodbus/datastore/store.py#L130
class TThreadSafeDataBlock(ModbusSequentialDataBlock):
  def __init__(self, address, values):
    super(TThreadSafeDataBlock, self).__init__(address, values)
    self.data_locker= threading.RLock()

  def start(self, address):
    return address - self.address

  def access_values(self, address, count=1):
    return super(TThreadSafeDataBlock, self).getValues(address, count)

  def getValues(self, address, count=1):
    #start= address - self.address
    #return self.values[start : start + count]
    with self.data_locker:
      return super(TThreadSafeDataBlock, self).getValues(address, count)

  def setValues(self, address, value):
    #if not isinstance(values, list):
      #values= [values]
    #start= address - self.address
    #self.values[start:start+len(values)]= values
    with self.data_locker:
      super(TThreadSafeDataBlock, self).setValues(address, value)

ADDR_ACTION_REQ= 1
ACTION_NONE= 0
ACTION_SET_ZERO= 1
ACTION_INCREMENT= 2
ACTION_ADD_INDEX= 3
ACTION_SQUARE= 4

class TDevice(object):
  def __init__(self):
    self.disc_in= TThreadSafeDataBlock(0, [False]*100)
    self.coil= TThreadSafeDataBlock(0, [False]*100)
    self.hold_reg= TThreadSafeDataBlock(0, [0]*100)
    self.input_reg= TThreadSafeDataBlock(0, [0]*100)

  def Storage(self):
    return dict(di=self.disc_in,
                co=self.coil,
                hr=self.hold_reg,
                ir=self.input_reg)

  def StartLoop(self):
    self.th= threading.Thread(name='loop', target=self.Loop)
    self.th.start()

  def Loop(self):
    print('''Keyboard operation:
      p: Print the registers.
      q: Quit the server.''')
    rate_adjuster= TRate(10)
    with TKBHit() as kbhit:
      while True:
        if kbhit.IsActive():
          key= kbhit.KBHit()
        else:  break
        if key=='q':
          print('Exiting the server...')
          break
        elif key=='p':
          print('In reg: {}, Hold reg: {}'.format(self.input_reg.getValues(1,10), self.hold_reg.getValues(0,3)))

        with self.hold_reg.data_locker:
          act_req= self.hold_reg.access_values(ADDR_ACTION_REQ, 1)[0]
          self.hold_reg.values[self.hold_reg.start(ADDR_ACTION_REQ)]= ACTION_NONE
        print('act_req:', act_req)

        if act_req==ACTION_SET_ZERO:
          with self.input_reg.data_locker:
            data= self.input_reg.access_values(1,10)
          data= [0]*10
          with self.input_reg.data_locker:
            start= self.input_reg.start(1)
            self.input_reg.values[start:start+10]= data
        elif act_req==ACTION_INCREMENT:
          with self.input_reg.data_locker:
            data= self.input_reg.access_values(1,10)
          data= [v+1 for v in data]
          with self.input_reg.data_locker:
            start= self.input_reg.start(1)
            self.input_reg.values[start:start+10]= data
        elif act_req==ACTION_ADD_INDEX:
          with self.input_reg.data_locker:
            data= self.input_reg.access_values(1,10)
          data= [v+i for i,v in enumerate(data)]
          with self.input_reg.data_locker:
            start= self.input_reg.start(1)
            self.input_reg.values[start:start+10]= data
        elif act_req==ACTION_SQUARE:
          with self.input_reg.data_locker:
            data= self.input_reg.access_values(1,10)
          data= [v*v for v in data]
          with self.input_reg.data_locker:
            start= self.input_reg.start(1)
            self.input_reg.values[start:start+10]= data

        rate_adjuster.sleep()
    print('End of TDevice.Loop')
    sys.exit(0)

def StartModbusServer(device, port):
  store= ModbusSlaveContext(**device.Storage())

  context= ModbusServerContext(slaves=store, single=True)

  identity= ModbusDeviceIdentification()
  identity.VendorName= 'AY'
  identity.ProductCode= 'AY'
  identity.VendorUrl= 'http://github.com/akihikoy/'
  identity.ProductName= 'Modbus Server'
  identity.ModelName= 'Test 3'
  identity.MajorMinorRevision= '0.1.0'

  StartTcpServer(context, identity=identity, address=('', port))


if __name__=='__main__':
  port= int(sys.argv[1]) if len(sys.argv)>1 else 5020
  device= TDevice()
  device.StartLoop()
  StartModbusServer(device, port)
  print('main thread has ended')

