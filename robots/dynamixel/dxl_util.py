#!/usr/bin/python
#\file    dxl_util.py
#\brief   Library to control a Dynamixel servo
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Mar.28, 2017
#\version 0.2
#\date    Aug, 2018
#         Implemented TDynamixelPortHandler for automatic reopening the port.
#\version 0.3
#\date    Nov.20, 2018
#         Added RH-P12-RN (Thormang3 gripper) protocol.

#This code is based on DynamixelSDK/python/protocol2_0/sync_read_write.py
#DynamixelSDK: https://github.com/ROBOTIS-GIT/DynamixelSDK
#Dynamixel XM430-W350: http://support.robotis.com/en/product/actuator/dynamixel_x/xm_series/xm430-w350.htm
#Dynamixel XH430-V350: http://support.robotis.com/en/product/actuator/dynamixel_x/xh_series/xh-430-v350.htm

import dxl_funcs as dynamixel  #Using Dynamixel SDK
import math, time, threading

class TDynamixelPortHandler(object):
  def __init__(self):
    #Callback in the reopen thread.  If this function returns False, the thread ends.
    self.ReopenCallback= None

    #List of {'dev':DEVICE_NAME, 'port_num':PORT_NUMBER, 'baudrate':BAUD_RATE}
    self.opened= []
    #Reopen thread state:
    self.thread_reopen= [False,None]

  #Return port number from device name.
  def Port(self, dev):
    try:
      return next(value for value in self.opened if value['dev']==dev)['port_num']
    except StopIteration:  return None

  #Open the device dev and set the baudrate if baudrate is not None.
  #If baudrate is None, reopen is True, and baudrate is saved in self.opened, it is set again.
  #Return the port number if succeeded, and None if failrue.
  #This will do nothing if the device is already opened (just returns the port number).
  def Open(self, dev='/dev/ttyUSB0', baudrate=None, reopen=False):
    set_baudrate= False
    try:
      #Check if dev is already opened.
      info= next(value for value in self.opened if value['dev']==dev)
      port_num= info['port_num']
      if all((baudrate is None, reopen, info['baudrate']>0)):
        baudrate= info['baudrate']
        #info['baudrate']= 0
        set_baudrate= True

    except StopIteration:
      info= {'port_num':None, 'dev':dev, 'baudrate':0}
      self.opened.append(info)

    if info['port_num'] is None:
      # Initialize PortHandler Structs
      # Set the port path
      # Get methods and members of PortHandlerLinux or PortHandlerWindows
      port_num= dynamixel.portHandler(dev)
      # Initialize PacketHandler Structs
      dynamixel.packetHandler()

      #Open port
      if dynamixel.openPort(port_num):
        print 'Opened a port:', dev, port_num
      else:
        print 'Failed to open a port:', dev, port_num
        dynamixel.closePort(port_num)
        return None
      info['port_num']= port_num

    if (baudrate is not None and info['baudrate']!=baudrate) or set_baudrate:
      #Set port baudrate
      if dynamixel.setBaudRate(port_num, int(baudrate)):
        print 'Changed the baud rate:', dev, port_num, baudrate
      else:
        print 'Failed to change the baud rate to:', dev, port_num, baudrate
        return None
      info['baudrate']= baudrate

    return port_num

  #Close the port.  Specify the port by device name (dev) OR port number (port_num).
  def Close(self, dev=None, port_num=None):
    i,info= -1,None
    pnum= None
    if dev is not None:
      try:
        i,info= next((i,value) for i,value in enumerate(self.opened) if value['dev']==dev)
        pnum= info['port_num']
      except StopIteration:
        pass
    elif port_num is not None:
      try:
        i,info= next((i,value) for i,value in enumerate(self.opened) if value['port_num']==port_num)
        pnum= port_num
      except StopIteration:
        pass
    #else:
      #raise Exception('TDynamixelPortHandler.Close: dev or port must be specified.')
    if pnum is not None:
      dynamixel.closePort(pnum)
      print 'Closed the port:',info['dev'],pnum
    if i>=0:  del self.opened[i]

  #Mark the device or port to be an error state.
  #Specify the port by device name (dev) OR port number (port_num).
  def MarkError(self, dev=None, port_num=None):
    info= None
    pnum= None
    if dev is not None:
      try:
        info= next(value for value in self.opened if value['dev']==dev)
        pnum= info['port_num']
      except StopIteration:
        pass
    elif port_num is not None:
      try:
        info= next(value for value in self.opened if value['port_num']==port_num)
        pnum= port_num
      except StopIteration:
        pass
    #else:
      #raise Exception('TDynamixelPortHandler.MarkError: dev or port must be specified.')
    if pnum is not None:
      dynamixel.closePort(pnum)
      print 'Closed the port:',info['dev'],pnum
    if info is not None:  info['port_num']= None

  #Start reopen thread to reopen all devices whose port_num is None.
  def StartReopen(self, interval=1.0):
    self.StopReopen()
    th_func= lambda:self.ReopenLoop(interval)
    self.thread_reopen= [True, threading.Thread(name='Reopen', target=th_func)]
    self.thread_reopen[1].start()

  #Stop reopen thread.
  def StopReopen(self):
    if self.thread_reopen[0]:
      self.thread_reopen[0]= False
      self.thread_reopen[1].join()
    self.thread_reopen= [False,None]

  #Reopen thread.
  #NOTE: Don't call this function directly.  Use self.StartReopen
  def ReopenLoop(self, interval):
    success= False
    while self.thread_reopen[0] and not success:
      if self.ReopenCallback is not None:
        if not self.ReopenCallback():  break
      success= True
      for info in self.opened:
        if info['port_num'] is not None:  continue
        port_num= self.Open(dev=info['dev'], reopen=True)
        if port_num is None:  success= False
      time.sleep(interval)
    self.thread_reopen[0]= False

#Global object:
DxlPortHandler= TDynamixelPortHandler()


class TDynamixel1(object):
  def __init__(self, type, dev='/dev/ttyUSB0'):
    self.WriteFuncs= (None, dynamixel.write1ByteTxRx, dynamixel.write2ByteTxRx, None, dynamixel.write4ByteTxRx)
    self.ReadFuncs= (None, dynamixel.read1ByteTxRx, dynamixel.read2ByteTxRx, None, dynamixel.read4ByteTxRx)

    # For Dynamixel PRO 54-200
    if type=='PRO54-200':  #FIXME: Correct name?
      #ADDR[NAME]=(ADDRESS,SIZE)
      self.ADDR={
        'TORQUE_ENABLE'       : (562,None),
        'GOAL_POSITION'       : (596,None),
        'PRESENT_POSITION'    : (611,None),
        }
      self.PROTOCOL_VERSION = 2  # Protocol version of Dynamixel

    # For Dynamixel XM430-W350
    elif type in ('XM430-W350','XH430-V350'):
      #ADDR[NAME]=(ADDRESS,SIZE)
      self.ADDR={
        'MODEL_NUMBER'        : (0,2),
        'MODEL_INFORMATION'   : (2,4),
        'FIRMWARE_VERSION'    : (6,1),
        'ID'                  : (7,1),
        'BAUD_RATE'           : (8,1),
        'RETURN_DELAY_TIME'   : (9,1),
        'OPERATING_MODE'      : (11,1),
        'TEMP_LIMIT'          : (31,1),
        'MAX_VOLT_LIMIT'      : (32,2),
        'MIN_VOLT_LIMIT'      : (34,2),
        'PWM_LIMIT'           : (36,2),
        'CURRENT_LIMIT'       : (38,2),
        'ACC_LIMIT'           : (40,4),
        'VEL_LIMIT'           : (44,4),
        'MAX_POS_LIMIT'       : (48,4),
        'MIN_POS_LIMIT'       : (52,4),
        'SHUTDOWN'            : (63,1),
        'TORQUE_ENABLE'       : (64,1),
        'HARDWARE_ERR_ST'     : (70,1),
        'VEL_I_GAIN'          : (76,2),
        'VEL_P_GAIN'          : (78,2),
        'POS_D_GAIN'          : (80,2),
        'POS_I_GAIN'          : (82,2),
        'POS_P_GAIN'          : (84,2),
        'GOAL_PWM'            : (100,2),
        'GOAL_CURRENT'        : (102,2),
        'GOAL_VELOCITY'       : (104,4),
        'GOAL_POSITION'       : (116,4),
        'PRESENT_PWM'         : (124,2),
        'PRESENT_CURRENT'     : (126,2),
        'PRESENT_VELOCITY'    : (128,4),
        'PRESENT_POSITION'    : (132,4),
        'PRESENT_IN_VOLT'     : (144,2),
        'PRESENT_TEMP'        : (146,1),
        }
      self.PROTOCOL_VERSION = 2  # Protocol version of Dynamixel

    # For RH-P12-RN
    elif type in ('RH-P12-RN',):
      #ADDR[NAME]=(ADDRESS,SIZE)
      self.ADDR={
        'MODEL_NUMBER'        : (0,2),
        'MODEL_INFORMATION'   : (2,4),
        'FIRMWARE_VERSION'    : (6,1),
        'ID'                  : (7,1),
        'BAUD_RATE'           : (8,1),
        'RETURN_DELAY_TIME'   : (9,1),
        'OPERATING_MODE'      : (11,1),
        'TEMP_LIMIT'          : (21,1),
        'MAX_VOLT_LIMIT'      : (22,2),
        'MIN_VOLT_LIMIT'      : (24,2),
        'PWM_LIMIT'           : (None,None),
        'CURRENT_LIMIT'       : (30,2),
        'ACC_LIMIT'           : (26,4),
        'VEL_LIMIT'           : (32,4),
        'MAX_POS_LIMIT'       : (36,4),
        'MIN_POS_LIMIT'       : (40,4),
        'SHUTDOWN'            : (48,1),
        'TORQUE_ENABLE'       : (562,1),
        'HARDWARE_ERR_ST'     : (892,1),
        'VEL_I_GAIN'          : (None,None),
        'VEL_P_GAIN'          : (None,None),
        'POS_D_GAIN'          : (590,2),
        'POS_I_GAIN'          : (592,2),
        'POS_P_GAIN'          : (594,2),
        'GOAL_PWM'            : (None,None),
        'GOAL_CURRENT'        : (604,2),
        'GOAL_VELOCITY'       : (600,4),
        'GOAL_POSITION'       : (596,4),
        'PRESENT_PWM'         : (None,None),
        'PRESENT_CURRENT'     : (621,2),
        'PRESENT_VELOCITY'    : (615,4),
        'PRESENT_POSITION'    : (611,4),
        'PRESENT_IN_VOLT'     : (623,2),
        'PRESENT_TEMP'        : (625,1),
        }
      self.PROTOCOL_VERSION = 2  # Protocol version of Dynamixel

    # Operation modes
    self.OP_MODE={
      'CURRENT'    : 0,   # Current Control Mode
      'VELOCITY'   : 1,   # Velocity Control Mode
      'POSITION'   : 3,   # Position Control Mode (default mode)
      'EXTPOS'     : 4,   # Extended Position Control Mode (Multi-turn)
      'CURRPOS'    : 5,   # Current-based Position Control Mode
      'PWM'        : 16,  # PWM Control Mode (Voltage Control Mode)
      }
    #NOTE: with 'RH-P12-RN', only CURRENT and CURRPOS are available.

    # Baud rates
    self.BAUD_RATE= [9600,57600,115200,1e6,2e6,3e6,4e6,4.5e6]
    if type=='RH-P12-RN':  self.BAUD_RATE.append(10.5e6)

    self.TORQUE_ENABLE  = 1  # Value for enabling the torque
    self.TORQUE_DISABLE = 0  # Value for disabling the torque
    self.MIN_POSITION = 0  # Dynamixel will rotate between this value
    self.MAX_POSITION = 4095  # and this value (note that the Dynamixel would not move when the position value is out of movable range. Check e-manual about the range of the Dynamixel you use.)
    if type=='RH-P12-RN':  self.MAX_POSITION = 1150
    if type=='XM430-W350':    self.MAX_CURRENT = 1193   # == Current Limit(38)
    elif type=='XH430-V350':  self.MAX_CURRENT = 689   # == Current Limit(38)
    elif type=='RH-P12-RN':   self.MAX_CURRENT = 820
    self.MAX_PWM = 885
    if type=='RH-P12-RN':  self.MAX_PWM = None

    self.POSITION_UNIT= math.pi/2048.0
    self.POSITION_OFFSET= 2048.0
    if type=='XM430-W350':
      self.CURRENT_UNIT= 2.69
      self.VELOCITY_UNIT= 0.229*(2.0*math.pi)/60.0
    elif type=='XH430-V350':
      self.CURRENT_UNIT= 1.34
      self.VELOCITY_UNIT= 0.229*(2.0*math.pi)/60.0
    elif type=='RH-P12-RN':
      self.CURRENT_UNIT= 4.02
      self.VELOCITY_UNIT= 0.114*(2.0*math.pi)/60.0

    self.COMM_SUCCESS = 0      # Communication Success result value
    self.COMM_TX_FAIL = -1001  # Communication Tx Failed

    # Configurable variables
    self.Id= 1  # Dynamixel ID
    self.Baudrate= 57600
    #self.Baudrate= 1000000
    self.DevName= dev  # Port name.  e.g. Win:'COM1', Linux:'/dev/ttyUSB0'
    self.OpMode= 'POSITION'
    if type=='RH-P12-RN':  self.OpMode= 'CURRPOS'
    self.CurrentLimit= self.MAX_CURRENT
    '''Shutdown mode:
      0x01: Input Voltage Error
      0x04: Overheating Error
      0x08: Motor Encoder Error
      0x10: Electrical Shock Error
      0x20: Overload Error '''
    self.Shutdown= 0x04+0x10+0x20  #Default
    #self.Shutdown= 0x04+0x10  #Ignoring Overload Error (DANGER mode but more powerful)

    self.GoalThreshold= 20

    # Status variables
    self.port_num= lambda self=self:DxlPortHandler.Port(self.DevName)
    self.dxl_result= None
    self.dxl_err= None

    # NOTE: We do not implement a thread lock for this low-level device controller.
    # Implement this functionality with the higher-level controller.
    #self.port_locker= threading.RLock()

  def __del__(self):
    self.Quit()


  #Conversion from Dynamixel PWM value to PWM(percentage).
  def ConvPWM(self,value):
    return value*100.0/self.MAX_PWM
  #Conversion from PWM(percentage) to Dynamixel PWM value.
  def InvConvPWM(self,value):
    return int(value*self.MAX_PWM/100.0)

  #Conversion from Dynamixel current value to current(mA).
  def ConvCurr(self,value):
    return value*self.CURRENT_UNIT
  #Conversion from current(mA) to Dynamixel current value.
  def InvConvCurr(self,value):
    return int(value/self.CURRENT_UNIT)

  #Conversion from Dynamixel velocity value to velocity(rad/s).
  def ConvVel(self,value):
    return value*self.VELOCITY_UNIT
  #Conversion from velocity(rad/s) to Dynamixel velocity value.
  def InvConvVel(self,value):
    return int(value/self.VELOCITY_UNIT)

  #Conversion from Dynamixel position value to position(rad).
  def ConvPos(self,value):
    return (value-self.POSITION_OFFSET)*self.POSITION_UNIT
  #Conversion from position(rad) to Dynamixel position value.
  def InvConvPos(self,value):
    return int(value/self.POSITION_UNIT+self.POSITION_OFFSET)

  #Conversion from Dynamixel temperature value to temperature(deg of Celsius).
  def ConvTemp(self,value):
    return value
  #Conversion from temperature(deg of Celsius) to Dynamixel temperature value.
  def InvConvTemp(self,value):
    return int(value)


  def Write(self, address, value):
    addr,size= self.ADDR[address]
    if addr is None:
      print '{address} is not available with this Dynamixel.'.format(address=address)
      return
    self.WriteFuncs[size](self.port_num(), self.PROTOCOL_VERSION, self.Id, addr, value)

  def Read(self, address):
    addr,size= self.ADDR[address]
    if addr is None:
      print '{address} is not available with this Dynamixel.'.format(address=address)
      return None
    value= self.ReadFuncs[size](self.port_num(), self.PROTOCOL_VERSION, self.Id, addr)
    if size==2:
      #value= value & 255
      #if value>127:  value= -(256-value)
      #value= value & 1023
      #if value>511:  value= -(1024-value)
      #value= value & 2047
      #if value>1023:  value= -(2048-value)
      #value= value & 4095
      #if value>2047:  value= -(4096-value)
      value= value & 65535
      if value>32767:  value= -(65536-value)
    return value

  def Setup(self):
    DxlPortHandler.Open(dev=self.DevName, baudrate=self.Baudrate)
    if self.port_num() is None:
      #self.port_num = None
      return False

    self.SetOpMode(self.OP_MODE[self.OpMode])
    #self.EnableTorque()
    return True

  def Quit(self):
    #self.DisableTorque()
    # Close port
    DxlPortHandler.Close(dev=self.DevName)
    #self.port_num = None

  #Check the result of sending a command.
  #Print the error message if quiet is False.
  #If reopen is True, we start the port reopen thread.
  def CheckTxRxResult(self, quiet=False, reopen=True):
    self.dxl_result= dynamixel.getLastTxRxResult(self.port_num(), self.PROTOCOL_VERSION)
    self.dxl_err= dynamixel.getLastRxPacketError(self.port_num(), self.PROTOCOL_VERSION)
    normal= self.dxl_result==self.COMM_SUCCESS and self.dxl_err==0
    if not normal:# and not quiet:
      print 'dxl_result=',self.dxl_result, 'dxl_err=',self.dxl_err
      if self.dxl_result != self.COMM_SUCCESS:
        print dynamixel.getTxRxResult(self.PROTOCOL_VERSION, self.dxl_result)
      elif self.dxl_err != 0:
        print dynamixel.getRxPacketError(self.PROTOCOL_VERSION, self.dxl_err)
    if not normal:
      DxlPortHandler.MarkError(dev=self.DevName)
      #self.port_num= None
      if reopen:  DxlPortHandler.StartReopen()
    return normal

  #Changing operating mode
  def SetOpMode(self, mode):
    self.DisableTorque()  #NOTE: We need to disable before changing the operating mode
    self.Write('OPERATING_MODE', mode)
    self.CheckTxRxResult()
    self.Write('CURRENT_LIMIT', self.CurrentLimit)
    self.Write('PWM_LIMIT', self.MAX_PWM)
    #self.Write('POS_I_GAIN', 16383)
    #self.Write('VEL_I_GAIN', 500)
    #self.Write('VEL_P_GAIN', 50)
    #self.Write('POS_D_GAIN', 2000)
    self.Write('SHUTDOWN', self.Shutdown)
    self.CheckTxRxResult()

    #self.Write('POS_D_GAIN', 2600)
    #self.Write('POS_I_GAIN', 1000)
    #self.Write('POS_P_GAIN', 800)
    self.CheckTxRxResult()
    #self.PrintStatus()

  #Enable Dynamixel Torque
  def EnableTorque(self):
    self.Write('TORQUE_ENABLE', self.TORQUE_ENABLE)
    if not self.CheckTxRxResult():  return False
    print 'Torque enabled'
    return True

  #Disable Dynamixel Torque
  def DisableTorque(self):
    self.Write('TORQUE_ENABLE', self.TORQUE_DISABLE)
    if not self.CheckTxRxResult():  return False
    print 'Torque disabled'
    return True

  #Print status
  def PrintStatus(self):
    status= []
    for address in self.ADDR.iterkeys():
      addr= self.ADDR[address][0]
      value= self.Read(address)
      if not self.CheckTxRxResult():  value= (value,'(error)')
      status.append([addr,address,value])
    status.sort()
    for addr,address,value in status:
      print '{address}({addr}): {value}'.format(address=address, addr=addr, value=value)
      #print '{address}({addr}) can not be observed.'.format(address=address, addr=addr)

  #Print shutdown description
  def print_shutdown(self, value, msg):
    if value & 0x01:  print '{0}: Input Voltage Error'.format(msg)
    if value & 0x04:  print '{0}: Overheating Error'.format(msg)
    if value & 0x08:  print '{0}: Motor Encoder Error'.format(msg)
    if value & 0x10:  print '{0}: Electrical Shock Error'.format(msg)
    if value & 0x20:  print '{0}: Overload Error'.format(msg)

  #Print shutdown configuration
  def PrintShutdown(self):
    self.print_shutdown(self.Read('SHUTDOWN'), 'SHUTDOWN')

  #Print hardware error status
  def PrintHardwareErrSt(self):
    self.print_shutdown(self.Read('HARDWARE_ERR_ST'), 'HARDWARE_ERR_ST')

  #Get current PWM
  def PWM(self):
    value= self.Read('PRESENT_PWM')
    self.CheckTxRxResult()
    return value

  #Get current current
  def Current(self):
    value= self.Read('PRESENT_CURRENT')
    self.CheckTxRxResult(quiet=True)
    return value

  #Get current velocity
  def Velocity(self):
    value= self.Read('PRESENT_VELOCITY')
    self.CheckTxRxResult(quiet=True)
    return value

  #Get current position
  def Position(self):
    value= self.Read('PRESENT_POSITION')
    self.CheckTxRxResult(quiet=True)
    return value

  #Get current temperature
  def Temperature(self):
    value= self.Read('PRESENT_TEMP')
    self.CheckTxRxResult(quiet=True)
    return value

  #Reboot Dynamixel
  def Reboot(self):
    dynamixel.reboot(self.port_num(), self.PROTOCOL_VERSION, self.Id)
    self.CheckTxRxResult()

  #Move the position to a given value.
  #  target: Target position, should be in [self.MIN_POSITION, self.MAX_POSITION]
  #  blocking: True: this function waits the target position is reached.  False: this function returns immediately.
  def MoveTo(self, target, blocking=True):
    target = int(target)
    #FIXME: If OpMode allows multi turn, target could vary.
    if target < self.MIN_POSITION:  target = self.MIN_POSITION
    elif target > self.MAX_POSITION:  target = self.MAX_POSITION

    # Write goal position
    self.Write('GOAL_POSITION', target)
    #if not self.CheckTxRxResult():  return
    self.CheckTxRxResult()
    #print 'debug:TDynamixel1:MoveTo:',target

    while blocking:
      pos= self.Position()
      if pos is None:  return
      #print("[ID:%03d] GoalPos:%03d  PresPos:%03d" % (self.Id, target, pos))
      if not (abs(target - pos) > self.GoalThreshold):  break

  #Move the position to a given value with given current.
  #  target: Target position, should be in [self.MIN_POSITION, self.MAX_POSITION]
  #  current: Target current, should be in [-self.MAX_CURRENT, self.MAX_CURRENT]
  #  blocking: True: this function waits the target position is reached.  False: this function returns immediately.
  def MoveToC(self, target, current, blocking=True):
    target= int(target)
    current= int(current)
    #FIXME: If OpMode allows multi turn, target could vary.
    if target < self.MIN_POSITION:  target = self.MIN_POSITION
    elif target > self.MAX_POSITION:  target = self.MAX_POSITION
    if current < -self.MAX_CURRENT:  current = -self.MAX_CURRENT
    elif current > self.MAX_CURRENT:  current = self.MAX_CURRENT

    # Write goal current and position
    self.Write('GOAL_CURRENT', current)
    #if not self.CheckTxRxResult():  return
    self.Write('GOAL_POSITION', target)
    #if not self.CheckTxRxResult():  return
    self.CheckTxRxResult()

    while blocking:
      pos= self.Position()
      if pos is None:  return
      #print("[ID:%03d] GoalPos:%03d  PresPos:%03d" % (self.Id, target, pos))
      if not (abs(target - pos) > self.GoalThreshold):  break

  #Set current
  def SetCurrent(self, current):
    if current < -self.MAX_CURRENT:  current = -self.MAX_CURRENT
    elif current > self.MAX_CURRENT:  current = self.MAX_CURRENT

    self.Write('GOAL_CURRENT', current)
    #if not self.CheckTxRxResult():  return
    self.CheckTxRxResult()

  #Set PWM
  def SetPWM(self, pwm):
    if pwm < -self.MAX_PWM:  pwm = -self.MAX_PWM
    elif pwm > self.MAX_PWM:  pwm = self.MAX_PWM

    self.Write('GOAL_PWM', pwm)
    #if not self.CheckTxRxResult():  return
    self.CheckTxRxResult()


if __name__=='__main__':
  import os
  import sys, tty, termios
  fd = sys.stdin.fileno()
  old_settings = termios.tcgetattr(fd)
  def getch():
    try:
      tty.setraw(sys.stdin.fileno())
      ch = sys.stdin.read(1)
    finally:
      termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

  dxl = TDynamixel1('XM430-W350')
  dxl.Setup()
  dxl.EnableTorque()

  index = 0
  positions = [dxl.MIN_POSITION, dxl.MAX_POSITION]
  while 1:
    print("Press any key to continue! (or press ESC to quit!)")
    if getch() == chr(0x1b):
      break

    dxl.MoveTo(positions[index])
    print 'Current position=',dxl.Position()
    index = 1 if index==0 else 0


