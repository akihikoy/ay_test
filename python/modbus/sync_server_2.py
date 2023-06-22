#!/usr/bin/python
#\file    sync_server_2.py
#\brief   Synchronous Modbus TCP server test 2.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jun.22, 2023
from pymodbus.server.sync import StartTcpServer
from pymodbus.device import ModbusDeviceIdentification
from pymodbus.datastore import ModbusSequentialDataBlock, ModbusSparseDataBlock
from pymodbus.datastore import ModbusSlaveContext, ModbusServerContext


#ref. ModbusSequentialDataBlock:
#https://github.com/pymodbus-dev/pymodbus/blob/2be46d/pymodbus/datastore/store.py#L130
class TCallbackDataBlock(ModbusSequentialDataBlock):
  def __init__(self, address, values, on_get_callback=None, on_set_callback=None):
    super(TCallbackDataBlock, self).__init__(address, values)
    self.on_get_callback= on_get_callback
    self.on_set_callback= on_set_callback

  def getValues(self, address, count=1):
    #start= address - self.address
    #return self.values[start : start + count]
    if self.on_get_callback is not None:
      self.on_get_callback(address, count)
    return super(TCallbackDataBlock, self).getValues(address, count)

  def setValues(self, address, value):
    #if not isinstance(values, list):
      #values= [values]
    #start= address - self.address
    #self.values[start:start+len(values)]= values
    super(TCallbackDataBlock, self).setValues(address, value)
    if self.on_set_callback is not None:
      self.on_set_callback(address, value)

def OnGetCallback(kind, address, count):
  print 'Get {}[{}:{}]'.format(kind, address, address+count)

def OnSetCallback(kind, address, value):
  print 'Set {}[{}] to {}'.format(kind, address, value)

def StartModbusServer():
  #store= ModbusSlaveContext(
      #di=ModbusSequentialDataBlock(0, [17]*100),
      #co=ModbusSequentialDataBlock(0, [17]*100),
      #hr=ModbusSequentialDataBlock(0, [17]*100),
      #ir=ModbusSequentialDataBlock(0, [17]*100))
  #store= ModbusSlaveContext(
      #di=TCallbackDataBlock(0, [17]*100),
      #co=TCallbackDataBlock(0, [17]*100),
      #hr=TCallbackDataBlock(0, [17]*100),
      #ir=TCallbackDataBlock(0, [17]*100))
  store= ModbusSlaveContext(
      di=TCallbackDataBlock(0, [17]*100, lambda address,count:OnGetCallback('diin',address,count), lambda address,value:OnSetCallback('diin',address,value)),
      co=TCallbackDataBlock(0, [17]*100, lambda address,count:OnGetCallback('coil',address,count), lambda address,value:OnSetCallback('coil',address,value)),
      hr=TCallbackDataBlock(0, [17]*100, lambda address,count:OnGetCallback('hreg',address,count), lambda address,value:OnSetCallback('hreg',address,value)),
      ir=TCallbackDataBlock(0, [17]*100, lambda address,count:OnGetCallback('ireg',address,count), lambda address,value:OnSetCallback('ireg',address,value)))

  context= ModbusServerContext(slaves=store, single=True)

  # ----------------------------------------------------------------------- #
  # initialize the server information
  # ----------------------------------------------------------------------- #
  # If you don't set this or any fields, they are defaulted to empty strings.
  # ----------------------------------------------------------------------- #
  identity= ModbusDeviceIdentification()
  identity.VendorName= 'AY'
  identity.ProductCode= 'AY'
  identity.VendorUrl= 'http://github.com/akihikoy/'
  identity.ProductName= 'Modbus Server'
  identity.ModelName= 'Test 2'
  identity.MajorMinorRevision= '0.1.0'

  StartTcpServer(context, identity=identity, address=('', 5020))


if __name__=='__main__':
  StartModbusServer()

