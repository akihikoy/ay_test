#!/usr/bin/python
#\file    cpsl08p1en_geh6060il_1.py
#\brief   Test code of operating the GEH6060IL gripper from the IO-Link master CPSL08P1EN through Modbus.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Sep.20, 2023
from __future__ import print_function
from pymodbus.client.sync import ModbusTcpClient as ModbusClient
from pymodbus.pdu import ExceptionResponse
from kbhit2 import KBHAskGen, KBHAskYesNo
import copy
import time

SERVER_URI= '10.10.6.207'
PORT= 502

#Output process data format of GEH6060IL.
OUT_DATA_FORMAT= [
    ('Control_Word', 2),
    ('Device_Mode', 1),
    ('Workpiece_No', 1),
    ('Reserve', 1),
    ('Position_Tolerance', 1),
    ('Grip_Force', 1),
    ('Drive_Velocity', 1),
    ('Base_Position', 2),
    ('Shift_Position', 2),
    ('Teach_Position', 2),
    ('Work_Position', 2),
  ]
OUT_DATA_COUNT= sum(size for key,size in OUT_DATA_FORMAT)
def DataToRegisterValues(data):
  return [
    data['Control_Word'],
    256*data['Device_Mode']
    +data['Workpiece_No'],
    256*data['Reserve']
    +data['Position_Tolerance'],
    256*data['Grip_Force']
    +data['Drive_Velocity'],
    data['Base_Position'],
    data['Shift_Position'],
    data['Teach_Position'],
    data['Work_Position'],
    ]

if __name__=='__main__':
  #Connection to the server:
  client= ModbusClient(SERVER_URI, port=PORT)
  client.connect()

  print('Connection established: {}'.format(client))

  in_data_address= 3002
  in_data_count= 3  #= 6 byte
  out_data_address= 4002

  def read():
    #Read registers:
    #res_r= client.read_input_registers(in_data_address, in_data_count)
    res_r= client.read_holding_registers(in_data_address, in_data_count)
    print('Read: {}'.format(res_r))
    if isinstance(res_r, ExceptionResponse):
      print('--Server trouble.')
    else:
      print('--Values: {}'.format(res_r.registers))

  def write(data):
    values= DataToRegisterValues(data)
    print('values: {}'.format(values))
    #KBHAskGen(' ')
    res_w= client.write_registers(out_data_address, values)
    print('Write: {}'.format(res_w))

  try:

    #Motor on:
    data_1= dict(
        Control_Word  =0,
        Device_Mode   =3,
        Workpiece_No  =0,
        Reserve       =0,
        Position_Tolerance=0,
        Grip_Force    =10,
        Drive_Velocity=10,
        Base_Position =100,
        Shift_Position=500,
        Teach_Position=0,
        Work_Position =1000)
    data_2= copy.deepcopy(data_1)
    data_2['Control_Word']= 1
    data_3= data_1

    #Motor off:
    data_11= copy.deepcopy(data_1)
    data_11['Device_Mode']= 5
    data_12= copy.deepcopy(data_11)
    data_12['Control_Word']= 1
    data_13= data_11

    #For gripper open close (position profile):
    data_p1= dict(
        Control_Word  =0,
        Device_Mode   =50,
        Workpiece_No  =0,
        Reserve       =0,
        Position_Tolerance=50,
        Grip_Force    =20,
        Drive_Velocity=100,
        Base_Position =100,
        Shift_Position=100,
        Teach_Position=2500,
        Work_Position =6000)
    data_p2= copy.deepcopy(data_p1)
    data_p2['Control_Word']= 1
    data_p3= data_p1
    data_p4= copy.deepcopy(data_p1)
    data_p4['Control_Word']= 512
    data_p5= copy.deepcopy(data_p1)
    data_p5['Control_Word']= 256

    #For gripper open close (force profile):
    data_f1= dict(
        Control_Word  =0,
        Device_Mode   =60,
        Workpiece_No  =0,
        Reserve       =0,
        Position_Tolerance=50,
        Grip_Force    =20,
        Drive_Velocity=100,
        Base_Position =100,
        Shift_Position=2000,
        Teach_Position=2500,
        Work_Position =3000)
    data_f2= copy.deepcopy(data_f1)
    data_f2['Control_Word']= 1
    data_f3= data_f1
    data_f4= copy.deepcopy(data_f1)
    data_f4['Control_Word']= 512
    data_f5= copy.deepcopy(data_f1)
    data_f5['Control_Word']= 256

    #For gripper open close (pre-pos force):
    data_pf1= dict(
        Control_Word  =0,
        Device_Mode   =80,
        Workpiece_No  =0,
        Reserve       =0,
        Position_Tolerance=50,
        Grip_Force    =20,
        Drive_Velocity=100,
        Base_Position =100,
        Shift_Position=2000,
        Teach_Position=2500,
        Work_Position =3000)
    data_pf2= copy.deepcopy(data_pf1)
    data_pf2['Control_Word']= 1
    data_pf3= data_pf1
    data_pf4= copy.deepcopy(data_pf1)
    data_pf4['Control_Word']= 512
    data_pf5= copy.deepcopy(data_pf1)
    data_pf5['Control_Word']= 256

    #For gripper open close with jog mode
    data_j1= dict(
        Control_Word  =0,
        Device_Mode   =11,
        Workpiece_No  =0,
        Reserve       =0,
        Position_Tolerance=10,
        Grip_Force    =20,
        Drive_Velocity=100,
        Base_Position =100,
        Shift_Position=2000,
        Teach_Position=2500,
        Work_Position =3000)
    data_j2= copy.deepcopy(data_j1)
    data_j2['Control_Word']= 1
    data_j3= data_j1
    data_j4= copy.deepcopy(data_j1)
    data_j4['Control_Word']= 1024
    data_j5= copy.deepcopy(data_j1)
    data_j5['Control_Word']= 2048
    data_j6= copy.deepcopy(data_j1)
    data_j6['Device_Mode']= 50
    data_j6['Control_Word']= 1

    #For gripper trajectory control (position profile):
    data_t1= dict(
        Control_Word  =0,
        Device_Mode   =50,
        Workpiece_No  =0,
        Reserve       =0,
        Position_Tolerance=50,
        Grip_Force    =20,
        Drive_Velocity=100,
        Base_Position =100,
        Shift_Position=200,
        Teach_Position=300,
        Work_Position =300)
    data_t2= copy.deepcopy(data_t1)
    data_t2['Control_Word']= 1
    #data_t3= data_t1
    data_traj= []
    for pos in [300,3000,2500,4000,2500,2600,2500,2600]:
      d1= copy.deepcopy(data_traj[-1] if len(data_traj)>0 else data_t1)
      d1['Control_Word']= 0
      d2= copy.deepcopy(d1)
      d2['Work_Position']= pos
      d2['Control_Word']= 1
      d3= copy.deepcopy(d2)
      d3['Control_Word']= 0
      d4= copy.deepcopy(d3)
      d4['Control_Word']= 512
      d5= copy.deepcopy(d4)
      d5['Control_Word']= 4
      data_traj.append(d1)
      data_traj.append(d2)
      data_traj.append(d3)
      data_traj.append(d4)
      data_traj.append(d5)
    data_tx= copy.deepcopy(data_t1)
    data_tx['Control_Word']= 256

    while True:
      read()

      #Write registers:
      command_list={
        'q': ('quit', None),
        'r': ('read', None),
        'e': ('enable motor',  [data_1, data_2, data_3]),
        'd': ('disable motor', [data_11, data_12, data_13]),
        'p': ('gripper control [position profile]',  [data_p1, data_p2, data_p3, data_p4, data_p5]),
        'f': ('gripper control [force profile]',  [data_f1, data_f2, data_f3, data_f4, data_f5]),
        'g': ('gripper control [pre-pos force]',  [data_pf1, data_pf2, data_pf3, data_pf4, data_pf5]),
        'j': ('gripper control [jog]', [data_j1, data_j2, data_j3, data_j4, data_j5, data_j6]),
        't': ('gripper control [trajectory]', [data_t1, data_t2]+data_traj+[data_tx]),
        }
      print('Type command:')
      for key,(help_str, data_seq) in command_list.items():
        print('  {}: {}'.format(key,help_str))
      key= KBHAskGen(*command_list.keys())
      if key=='q':
        break
      elif key=='r':
        read()

      else:
        help_str, data_seq= command_list[key]
        print('')
        print('###Executing the {} sequence.###'.format(help_str))
        print('Hit space to continue at each step.')
        for data in data_seq:
          write(data)
          read()
          KBHAskGen(' ')
          read()

  finally:
    #Disconnect from the server.*
    client.close()
