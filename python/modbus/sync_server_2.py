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

def MergeDict(d_base, d_new, allow_new_key=True):
  if isinstance(d_new, (list,tuple)):
    for d_new_i in d_new:
      MergeDict(d_base, d_new_i)
  else:
    for k_new,v_new in d_new.iteritems() if hasattr(d_new,'iteritems') else d_new.items():
      if not allow_new_key and k_new not in d_base:
        raise Exception('MergeDict: Unexpected key:',k_new)
      if k_new in d_base and (isinstance(v_new,dict) and isinstance(d_base[k_new],dict)):
        MergeDict(d_base[k_new], v_new)
      else:
        d_base[k_new]= v_new
  return d_base  #NOTE: d_base is overwritten. Returning it is for the convenience.

#ref. ModbusSequentialDataBlock:
#https://github.com/pymodbus-dev/pymodbus/blob/2be46d/pymodbus/datastore/store.py#L130
class TCallbackDataBlock(ModbusSequentialDataBlock):
  def __init__(self, address, values, callbacks=None):
    super(TCallbackDataBlock, self).__init__(address, values)
    default_callbacks= {e:lambda *args,**kwargs:None for e in
                        ('before_get', 'after_get',
                         'before_set', 'after_set')}
    self.callbacks= MergeDict(default_callbacks, callbacks, allow_new_key=False) if callbacks is not None else default_callbacks

  def access_values(self, address, count=1):
    return super(TCallbackDataBlock, self).getValues(address, count)

  def getValues(self, address, count=1):
    #start= address - self.address
    #return self.values[start : start + count]
    self.callbacks['before_get'](self, address, count)
    res= super(TCallbackDataBlock, self).getValues(address, count)
    self.callbacks['after_get'](self, address, count)
    return res

  def setValues(self, address, value):
    #if not isinstance(values, list):
      #values= [values]
    #start= address - self.address
    #self.values[start:start+len(values)]= values
    self.callbacks['before_set'](self, address, value)
    super(TCallbackDataBlock, self).setValues(address, value)
    self.callbacks['after_set'](self, address, value)

def BeforeGetCallback(kind, db, address, count):
  #print 'Get {}[{}:{}]'.format(kind, address, address+count)
  print 'Get {}[{}:{}] = {}'.format(kind, address, address+count, db.access_values(address,count))

def BeforeSetCallback(kind, db, address, value):
  #print 'Set {}[{}] to {}'.format(kind, address, value)
  val_orig= db.access_values(address,len(value) if isinstance(value,list) else 1)
  print 'Set {}[{}] from {} to {}'.format(kind, address, val_orig, value)

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
  def make_cb(kind):
    return {'before_get':lambda db,address,count:BeforeGetCallback(kind,db,address,count),
            'before_set':lambda db,address,value:BeforeSetCallback(kind,db,address,value)}
  store= ModbusSlaveContext(
      di=TCallbackDataBlock(0, [17]*100, make_cb('disc in')),
      co=TCallbackDataBlock(0, [17]*100, make_cb('coil')),
      hr=TCallbackDataBlock(0, [17]*100, make_cb('hold reg')),
      ir=TCallbackDataBlock(0, [17]*100, make_cb('in reg')))

  context= ModbusServerContext(slaves=store, single=True)

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

