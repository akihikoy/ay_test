#!/usr/bin/python
#\file    cont_io.py
#\brief   Continuous input scan/output from GPIO.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jan.22, 2024
from __future__ import print_function
import ctypes
import cdio
import sys
import time
from kbhit2 import TKBHit

if __name__=='__main__':
  dio_id= ctypes.c_short()
  #bit_no= ctypes.c_short()
  err_str= ctypes.create_string_buffer(256)

  dev_name= 'DIO000'
  lret= cdio.DioInit(dev_name.encode(), ctypes.byref(dio_id))
  if lret!=cdio.DIO_ERR_SUCCESS:
    cdio.DioGetErrorString(lret, err_str)
    print('Error in DioInit: {}: {}'.format(lret, err_str.value.decode('utf-8')))
    sys.exit()

  out_port_data= [[False]*8,[False]*8]
  with TKBHit() as kbhit:
    while kbhit.IsActive():
      cmd= kbhit.KBHit()
      if cmd is not None:
        if cmd=='q':  break
        else:
          i_port= int(cmd)
          if i_port<8:  out_port_data[0][i_port]= not out_port_data[0][i_port]
          else:         out_port_data[1][i_port-8]= not out_port_data[1][i_port-8]
        print(cmd)

      in_port_data= [None,None]
      for i_port in range(2):
        port_no= ctypes.c_short(int(i_port))
        in_data= ctypes.c_ubyte()
        lret= cdio.DioInpByte(dio_id, port_no, ctypes.byref(in_data))
        if lret==cdio.DIO_ERR_SUCCESS:
          #in_port_data[i_port]= in_data.value
          in_port_data[i_port]= [in_data.value&ibit!=0 for ibit in (1,2,4,8,16,32,64,128)]
        else:
          in_port_data[i_port]= None
          cdio.DioGetErrorString(lret, err_str)
          print('Error in DioInpByte: {}: {}'.format(lret, err_str.value.decode('utf-8')))

      for i_port in range(2):
        port_no= ctypes.c_short(int(i_port))
        out_data= ctypes.c_ubyte(sum(ibit if obit else 0 for obit,ibit in zip(out_port_data[i_port],(1,2,4,8,16,32,64,128))))
        lret= cdio.DioOutByte(dio_id, port_no, out_data)
        if lret!=cdio.DIO_ERR_SUCCESS:
          cdio.DioGetErrorString(lret, err_str)
          print('Error in DioOutByte: {}: {}'.format(lret, err_str.value.decode('utf-8')))

      #print('Input: 0:0x{:02x} 1:0x{:02x}'.format(in_port_data[0], in_port_data[1]))
      print('Input:  0:{} 1:{}'.format(in_port_data[0], in_port_data[1]))
      print('Output: 0:{} 1:{}'.format(out_port_data[0], out_port_data[1]))
      time.sleep(0.005)

  lret= cdio.DioExit(dio_id)
  if lret!=cdio.DIO_ERR_SUCCESS:
    cdio.DioGetErrorString(lret, err_str)
    print('Error in DioExit: {}: {}'.format(lret, err_str.value.decode('utf-8')))

