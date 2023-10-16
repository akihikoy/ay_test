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
import numpy as np

#IO-Link master CPSL08P1EN (Modbus) server info:
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

def BitsToBoolList(value, n_bits):
  return [bool(value&(1<<i)) for i in range(n_bits)]

STATUS_BIT_MEANINGS=[
    'Encoder:OK',
    'Motor:ON',
    'Gripper:Moving',
    'Gripper:Stop',
    'JogTo:Base',
    'JogTo:Work',
    'IO-Link:OK',
    'Controller:Trouble',
    'Position:Base',
    'Position:Teach',
    'Position:Work',
    'Position:Others',
    'DataTransfer:OK',
    'PositionReq:Base',
    'PositionReq:Work',
    'Error',
  ]
STATUS_STR_TO_BIT= {s:i for i,s in enumerate(STATUS_BIT_MEANINGS)}

def StatusWordToStr(status_word):
  bits= BitsToBoolList(status_word, n_bits=16)
  on_bits= np.where(bits)[0]
  return [STATUS_BIT_MEANINGS[idx] for idx in on_bits]

DIAGNOSIS_MEANINGS={
  0x0000: 'Device is ready for operation.',
  0x0001: 'Motor controller is switched off.',
  0x0100: 'Actuator power supply is not present or is too low.',
  0x0101: 'Temperature above maximum permitted temperature.',
  0x0102: 'Max. permitted temperature undershot.',
  0x0206: 'Motion task cannot be executed (CRC error).',
  0x0300: 'ControlWord is not plausible. Initial state after gripper restart',
  0x0301: 'Positions are not plausible.',
  0x0302: 'GripForce is not plausible.',
  0x0303: 'DriveVelocity is not plausible.',
  0x0304: 'PositionTolerance is not plausible.',
  0x0305: 'Position measuring system not referenced.',
  0x0306: 'DeviceMode is not plausible.',
  0x0307: 'Motion task cannot be executed.',
  0x0308: 'WorkpieceNo cannot be selected.',
  0x0313: 'Calculated ShiftPosition exceeded.',
  0x0402: 'Jam',
  0x0404: 'Position sensor error',
  0x0406: 'Internal error',
  0x040B: 'Internal error',
  0x040C: 'Internal error',
  0x040D: 'Internal error',
  0x040E: 'Internal error',
  0x040F: 'Internal error',
  }

def DiagnosisWordToStr(diagnosis_word):
  return DIAGNOSIS_MEANINGS[diagnosis_word] if diagnosis_word in DIAGNOSIS_MEANINGS else 'UNKNOWN'

if __name__=='__main__':
  #Connection to the server:
  client= ModbusClient(SERVER_URI, port=PORT)
  client.connect()

  print('Connection established: {}'.format(client))

  #IO-Link master CPSL08P1EN (Modbus) configuration:
  in_data_address= 3002
  in_data_count= 3  #= 6 byte
  out_data_address= 4002

  #Gripper configuration:
  #GRIPPER_TYPE= 'GEH6060IL-03-B'
  GRIPPER_TYPE= 'GEH6040IL-31-B'

  if GRIPPER_TYPE=='GEH6060IL-03-B':
    MAX_POSITION= 6000
    SELF_LOCK= True
  elif GRIPPER_TYPE=='GEH6040IL-31-B':
    MAX_POSITION= 4000
    SELF_LOCK= False

  def read():
    #Read registers:
    #res_r= client.read_input_registers(in_data_address, in_data_count)
    res_r= client.read_holding_registers(in_data_address, in_data_count)
    if isinstance(res_r, ExceptionResponse):
      print('Read: {}'.format(res_r))
      print('--Server trouble.')
      return None
    else:
      #print('--Values: {}'.format(res_r.registers))
      #registers: [Status Word(16bits), Diagnosis (2byte), Actual Position(2byte, unit=0.01mm).
      status_word, diagnosis, pos_curr= res_r.registers
      #Verbose print:
      #print('Read: {}'.format(res_r))
      #print('--Status: {}'.format(StatusWordToStr(res_r.registers[0])))
      #print('--Diagnosis: {}'.format(hex(res_r.registers[1])))
      #print('--Position: {} mm ({})'.format(0.01*res_r.registers[2], res_r.registers[2]))
      #Short print:
      print('  Read: Diag:{} Pos:{}mm ({}) St: {}'.format(hex(res_r.registers[1]), 0.01*res_r.registers[2], res_r.registers[2], StatusWordToStr(res_r.registers[0]) ))
      if res_r.registers[1]>0:
        print('    Diag: {}: {}'.format(hex(res_r.registers[1]), DiagnosisWordToStr(res_r.registers[1]) ))
      return status_word, diagnosis, pos_curr

  def write(data):
    values= DataToRegisterValues(data)
    #KBHAskGen(' ')
    res_w= client.write_registers(out_data_address, values)
    #Verbose print:
    #print('Write: {}'.format([data[key] for key,size in OUT_DATA_FORMAT]))
    #print('--Packets: {}'.format(values))
    #print('--Write: {}'.format(res_w))
    #Short print:
    print('Write: {}'.format([data[key] for key,size in OUT_DATA_FORMAT]))

  #Read IO-Link master status.
  def read_master_st():
    #Read registers:
    res_r= client.read_holding_registers(3000, 1)
    print('  Read: {}'.format(res_r))
    if isinstance(res_r, ExceptionResponse):
      print('  --Server trouble.')
      return None
    else:
      master_st_bits= BitsToBoolList(res_r.registers[0], n_bits=16)
      com_st, pd_valid_st= master_st_bits[0], master_st_bits[8]  #COM status (True=OK), PD (process data) valid status (True=OK).
      print('  --IO-Link Master Status: COM:{}, PD:{}'.format(com_st, pd_valid_st))
      return com_st, pd_valid_st

  #Wait for status[bit_name]==state.
  def wait_for_status(bit_name, state, dt_sleep):
    i_bit= STATUS_STR_TO_BIT[bit_name]
    while True:
      status_word,diagnosis,pos_curr= read()
      status_bits= BitsToBoolList(status_word, n_bits=16)
      if status_bits[i_bit]==state:  break
      if status_bits[-1]:
        print('''###Error during waiting for '{}':{}###'''.format(bit_name,state))
        read()
        return False
      if dt_sleep is not None: time.sleep(dt_sleep)
    return True

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
  data_seq_e= [data_1, data_2, data_3]

  #Motor off:
  data_11= copy.deepcopy(data_1)
  data_11['Device_Mode']= 5
  data_12= copy.deepcopy(data_11)
  data_12['Control_Word']= 1
  data_13= data_11
  data_seq_d= [data_11, data_12, data_13]

  #Outside homing:
  data_h1= dict(
      Control_Word  =0,
      Device_Mode   =10,
      Workpiece_No  =0,
      Reserve       =0,
      Position_Tolerance=50,
      Grip_Force    =20,
      Drive_Velocity=10,
      Base_Position =100,
      Shift_Position=100,
      Teach_Position=0,
      Work_Position =MAX_POSITION)
  data_h2= copy.deepcopy(data_h1)
  data_h2['Control_Word']= 1
  data_h3= data_h1
  data_seq_h1= [data_h1, data_h2, data_h3]

  #For gripper open close (position profile):
  data_p1= dict(
      Control_Word  =0,
      Device_Mode   =50,
      #Device_Mode   =51,
      Workpiece_No  =0,
      Reserve       =0,
      Position_Tolerance=50,
      Grip_Force    =20,
      Drive_Velocity=100,
      Base_Position =100,
      Shift_Position=100,
      Teach_Position=2500,
      Work_Position =MAX_POSITION)
  data_p2= copy.deepcopy(data_p1)
  data_p2['Control_Word']= 1
  data_p3= data_p1
  data_p4= copy.deepcopy(data_p1)
  data_p4['Control_Word']= 512
  data_p5= copy.deepcopy(data_p1)
  data_p5['Control_Word']= 256
  data_seq_p1= [data_p1, data_p2, data_p3, data_p4, data_p5]

  #For gripper open close (position profile) with ResetDirectionFlag:
  #NOTE: After the experiments, it turned out that the reset does not affect.
  data_p11= copy.deepcopy(data_p1)
  data_p12= copy.deepcopy(data_p11)
  data_p12['Control_Word']= 1
  data_p13= data_p11
  data_p14= copy.deepcopy(data_p11)
  data_p14['Control_Word']= 512
  data_p15= copy.deepcopy(data_p11)
  data_p15['Control_Word']= 4
  data_p16= copy.deepcopy(data_p11)
  data_p16['Control_Word']= 256
  data_seq_p2= [data_p11, data_p12, data_p13, data_p14, data_p15, data_p16]

  #Gripper close -> stop -> open (position profile).
  def close_stop_open1():
    data_p21= copy.deepcopy(data_p1)
    data_p21['Drive_Velocity']= 20
    data_p22= copy.deepcopy(data_p21)
    data_p22['Control_Word']= 1
    data_p23= data_p21
    data_p24= copy.deepcopy(data_p21)
    data_p24['Control_Word']= 512

    for data in [data_p21, data_p22, data_p23, data_p24]:
      write(data)
      read()
      KBHAskGen(' ')
      read()

    d= copy.deepcopy(data_p21)
    d['Control_Word']= 1  #Data transfer; NOTE: At this point, the gripper stops.
    write(d)
    read()
    KBHAskGen(' ')

    status_word,diagnosis,pos_curr= read()
    #d['Work_Position']= pos_curr  #Consider to change the work position to the current to stop, but it is not needed.
    #write(d)
    #read()
    #KBHAskGen(' ')
    #read()
    d['Control_Word']= 0  #Complete
    write(d)
    read()
    KBHAskGen(' ')
    read()

    data_p25= copy.deepcopy(d)
    data_p25['Control_Word']= 256
    write(data_p25)
    KBHAskGen(' ')

  #For gripper open close (force profile):
  data_f1= dict(
      Control_Word  =0,
      Device_Mode   =60 if SELF_LOCK else 62,
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
  data_seq_f1= [data_f1, data_f2, data_f3, data_f4, data_f5]

  #For gripper open close (pre-pos force):
  data_pf1= dict(
      Control_Word  =0,
      Device_Mode   =80 if SELF_LOCK else 82,
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
  data_seq_pf1= [data_pf1, data_pf2, data_pf3, data_pf4, data_pf5]

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
  data_seq_j1= [data_j1, data_j2, data_j3, data_j4, data_j5, data_j6]

  traj_1= [350,3000,2500,4000,2510,2600,2520,2610,2700,2800,2900,350]
  traj_1= [int(p*MAX_POSITION/6000) for p in traj_1]

  #For gripper trajectory control (position profile):
  data_t1= dict(
      Control_Word  =0,
      Device_Mode   =50,
      #Device_Mode   =51,
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
  for pos in traj_1:
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
  data_tx= copy.deepcopy(data_traj[-1])
  data_tx['Control_Word']= 256
  data_seq_t1= [data_t1, data_t2]+data_traj+[data_tx]

  #Following a trajectory
  def follow_traj1():
    data_t10= dict(
        Control_Word  =0,
        #Device_Mode   =50,
        Device_Mode   =51,  #Fast positioning mode.
        Workpiece_No  =0,
        Reserve       =0,
        Position_Tolerance=50,
        Grip_Force    =30,
        Drive_Velocity=100,
        Base_Position =75,
        Shift_Position=76,
        Teach_Position=0,
        Work_Position =300)
    dt_sleep= None
    #dt_sleep= 0.001
    #dt_sleep= 0.005
    def sleep():
      if dt_sleep is not None: time.sleep(dt_sleep)
    write(data_t10)
    sleep()
    d= copy.deepcopy(data_t10)
    for pos in traj_1:
      d['Control_Word']= 1  #Data transfer
      write(d)
      sleep()
      if not wait_for_status(bit_name='DataTransfer:OK', state=True, dt_sleep=dt_sleep):  return
      d['Work_Position']= pos
      write(d)
      #sleep()
      print('  >>>>><<<<Waiting for transmission complete>>>>><<<<')
      for i in range(5):  #5 is a magic number, but works.
        #com_st,pd_valid_st= read_master_st()  #IO-Link master state does not matter.
        read()
        sleep()
      #if not wait_for_status(bit_name='DataTransfer:OK', state=False, dt_sleep=dt_sleep):  return
      d['Control_Word']= 0  #Complete
      write(d)
      sleep()
      if not wait_for_status(bit_name='DataTransfer:OK', state=False, dt_sleep=dt_sleep):  return
      d['Control_Word']= 512  #MoveToWork
      write(d)
      sleep()
      if not wait_for_status(bit_name='Position:Work', state=True, dt_sleep=dt_sleep):  return
      #if not wait_for_status(bit_name='Gripper:Stop', state=True, dt_sleep=dt_sleep):  return
      #if not wait_for_status(bit_name='Gripper:Moving', state=True, dt_sleep=dt_sleep):  return
      #if not wait_for_status(bit_name='Gripper:Moving', state=False, dt_sleep=dt_sleep):  return
      d['Control_Word']= 4  #ResetDirectionFlag
      write(d)
      sleep()
      if not wait_for_status(bit_name='PositionReq:Work', state=False, dt_sleep=dt_sleep):  return

  #Following a trajectory ver.2
  def follow_traj2():
    data_t10= dict(
        Control_Word  =0,
        #Device_Mode   =50,
        Device_Mode   =51,  #Fast positioning mode.
        Workpiece_No  =0,
        Reserve       =0,
        Position_Tolerance=20,
        Grip_Force    =30,
        Drive_Velocity=100,
        Base_Position =75,
        Shift_Position=76,
        Teach_Position=0,
        Work_Position =300)
    dt_sleep= None
    #dt_sleep= 0.001
    #dt_sleep= 0.005
    def sleep():
      if dt_sleep is not None: time.sleep(dt_sleep)
    write(data_t10)
    sleep()
    d= copy.deepcopy(data_t10)
    pos_prev= read()[2]
    for pos in traj_1:
      d['Control_Word']= 1  #Data transfer
      write(d)
      sleep()
      if not wait_for_status(bit_name='DataTransfer:OK', state=True, dt_sleep=dt_sleep):  return
      d['Work_Position']= pos
      write(d)
      #sleep()
      print('  >>>>><<<<Waiting for transmission complete>>>>><<<<')
      for i in range(5):  #5 is a magic number, but works.
        #com_st,pd_valid_st= read_master_st()  #IO-Link master state does not matter.
        read()
        sleep()
      #if not wait_for_status(bit_name='DataTransfer:OK', state=False, dt_sleep=dt_sleep):  return
      d['Control_Word']= 0  #Complete
      write(d)
      sleep()
      if not wait_for_status(bit_name='DataTransfer:OK', state=False, dt_sleep=dt_sleep):  return
      d['Control_Word']= 512  #MoveToWork
      write(d)
      #sleep()
      gripper_speed= 80.0  #Data sheet gripper speed: 60.0 mm/s
      if GRIPPER_TYPE=='GEH6040IL-31-B':
        gripper_speed= 120.0  #Data sheet gripper speed: 120.0 mm/s
      dt_sleep2= abs(pos-pos_prev)*0.01/gripper_speed
      print('dt_sleep2: {}'.format(dt_sleep2))
      time.sleep(dt_sleep2)
      pos_prev= pos
      #if not wait_for_status(bit_name='Position:Work', state=True, dt_sleep=dt_sleep):  return
      d['Control_Word']= 1  #Data transfer (to stop)
      write(d)
      #sleep()
      if not GRIPPER_TYPE=='GEH6040IL-31-B':
        if not wait_for_status(bit_name='DataTransfer:OK', state=True, dt_sleep=dt_sleep):  return
      #time.sleep(0.15)
      d['Control_Word']= 4  #ResetDirectionFlag
      write(d)
      sleep()
      if not GRIPPER_TYPE=='GEH6040IL-31-B':
        if not wait_for_status(bit_name='PositionReq:Work', state=False, dt_sleep=dt_sleep):  return
      #time.sleep(0.05)

  #Jog motion with position profile.
  #  step: Step to move (plus or minus).  If zero, 4(ResetDirectionFlag) is sent.
  def jog_with_posp(step):
    pos_curr= read()[2]
    pos_trg= min(MAX_POSITION,max(100,pos_curr+step))
    data_t10= dict(
        Control_Word  =0,
        Device_Mode   =50,
        #Device_Mode   =51,  #Fast positioning mode.
        Workpiece_No  =0,
        Reserve       =0,
        Position_Tolerance=50,
        Grip_Force    =100,
        Drive_Velocity=100,
        Base_Position =75,
        Shift_Position=76,
        Teach_Position=0,
        Work_Position =pos_curr)
    dt_sleep= None
    def sleep():
      if dt_sleep is not None: time.sleep(dt_sleep)
    if step==0:
      data_t10['Control_Word']= 4  #ResetDirectionFlag
      write(data_t10)
      sleep()
      return
    write(data_t10)
    sleep()
    d= copy.deepcopy(data_t10)
    d['Control_Word']= 1  #Data transfer
    d['Work_Position']= pos_trg
    write(d)
    #sleep()
    if not wait_for_status(bit_name='DataTransfer:OK', state=True, dt_sleep=dt_sleep):  return
    d['Control_Word']= 0  #Complete
    write(d)
    sleep()
    if not wait_for_status(bit_name='DataTransfer:OK', state=False, dt_sleep=dt_sleep):  return
    d['Control_Word']= 512  #MoveToWork
    write(d)
    sleep()
    time.sleep(0.05)
    d['Control_Word']= 4  #ResetDirectionFlag
    write(d)
    sleep()

  #Jog motion with jog mode.
  #  step: +: close, -: open, 0: stop/start-jog.
  def jog_with_jog(step):
    data_j11= dict(
        Control_Word  =0,
        Device_Mode   =11,
        Workpiece_No  =0,
        Reserve       =0,
        Position_Tolerance=10,
        Grip_Force    =20,
        Drive_Velocity=100,
        Base_Position =75,
        Shift_Position=76,
        Teach_Position=100,
        Work_Position =MAX_POSITION)
    data_j12= copy.deepcopy(data_j11)
    data_j12['Control_Word']= 1
    data_j13= data_j11
    data_j14= copy.deepcopy(data_j11)
    data_j14['Control_Word']= 1024
    data_j15= copy.deepcopy(data_j11)
    data_j15['Control_Word']= 2048
    data_j16= copy.deepcopy(data_j11)
    data_j16['Device_Mode']= 50
    data_j16['Control_Word']= 1
    #data_seq_j1= [data_j1, data_j2, data_j3, data_j4, data_j5, data_j6]
    dt_sleep= None
    def sleep():
      if dt_sleep is not None: time.sleep(dt_sleep)
    if step==0:
      write(data_j11)
      sleep()
      write(data_j12)
      sleep()
      if not wait_for_status(bit_name='DataTransfer:OK', state=True, dt_sleep=dt_sleep):  return
      write(data_j13)
      sleep()
      if not wait_for_status(bit_name='DataTransfer:OK', state=False, dt_sleep=dt_sleep):  return
      return
    elif step>0:
      write(data_j14)
      time.sleep(abs(step))
      write(data_j12)
      return
    elif step<0:
      write(data_j15)
      time.sleep(abs(step))
      write(data_j12)
      return

  try:
    while True:
      read()

      #Write registers:
      command_list=[
        ['q', ('quit', None)],
        ['r', ('read', None)],
        ['E', ('enable motor',  data_seq_e)],
        ['D', ('disable motor', data_seq_d)],
        ['H', ('outside homing', data_seq_h1)],
        ['p', ('gripper control [position profile]',  data_seq_p1)],
        ['[', ('gripper control [position profile/with reset]',  data_seq_p2)],
        [']', ('gripper control [position profile/with stop]',  close_stop_open1)],
        ['f', ('gripper control [force profile]',  data_seq_f1)],
        ['g', ('gripper control [pre-pos force]',  data_seq_pf1)],
        ['j', ('gripper control [jog]', data_seq_j1)],
        ['t', ('gripper control [trajectory/step by step]', data_seq_t1)],
        ['y', ('gripper control [trajectory/auto]', follow_traj1)],
        ['u', ('gripper control [trajectory/auto2]', follow_traj2)],
        ['z', ('jog with position profile [+/close]', lambda: jog_with_posp(100))],
        ['x', ('jog with position profile [reset_dir]', lambda: jog_with_posp(0))],
        ['c', ('jog with position profile [-/open]', lambda: jog_with_posp(-100))],
        ['a', ('jog with jog mode [+/close]', lambda: jog_with_jog(0.1))],
        ['s', ('jog with jog mode [reset_dir]', lambda: jog_with_jog(0))],
        ['d', ('jog with jog mode [-/open]', lambda: jog_with_jog(-0.1))],
        ]
      command_dict= {key:(help_str, data_seq) for key,(help_str, data_seq) in command_list}
      print('Type command:')
      for key,(help_str, data_seq) in command_list:
        print('  {}: {}'.format(key,help_str))
      key= KBHAskGen(*command_dict.keys())
      t_start= time.time()
      help_str, data_seq= command_dict[key]
      print('#{}#'.format(help_str))
      if key=='q':
        break
      elif key=='r':
        read()
      elif isinstance(data_seq, list):
        print('')
        print('###Executing the {} sequence.###'.format(help_str))
        print('Hit space to continue at each step.')
        for data in data_seq:
          write(data)
          read()
          KBHAskGen(' ')
          read()
      elif callable(data_seq):
        data_seq()
      print('Task {}[{}] completed. Duration: {}s'.format(key, help_str, time.time()-t_start))

  finally:
    #Disconnect from the server.*
    client.close()
