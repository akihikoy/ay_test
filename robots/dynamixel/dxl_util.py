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
#\version 0.4
#\date    Apr.18, 2019
#         Modified for the latest SDK which no longer uses the shared object implemented in C.
#\version 0.5
#\date    Jun.11, 2019
#         Added MX-64AR protocol 2.0 for SAKE EZGripper Gen2.
#         Note: MX-64AR of EZGripper is originally protocol 1.0; update the firmware.
#\version 0.6
#\date    Jan.28, 2020
#         Added PH54-200-S500, XH540-W270.
#\version 0.7
#\date    Sep.30, 2021
#         Added PM54-060-S250.
#\version 0.8
#\date    May.19, 2022
#         Added XD540-T270.
#\version 0.9
#\date    Sep.19, 2022
#         Added RH-P12-RN(A).

#cf. DynamixelSDK/python/tests/protocol2_0/read_write.py
#DynamixelSDK: https://github.com/ROBOTIS-GIT/DynamixelSDK
#Control table comparison:
#  https://docs.google.com/spreadsheets/d/19Zlqyls2ZCFspiZLJ6MFhfvJ9L4j-N5LDj3l2gndB-c/edit#gid=0

import dynamixel_sdk as dynamixel  #Using Dynamixel SDK
import math, time, threading

'''Dynamixel port hander class.
+ It provides a function to automatically reopen a port.
+ It supports multiple port-open request.
+ This is a singleton class.  Use the global instance DxlPortHandler.'''
class TDynamixelPortHandler(object):
  #A most simplest singleton:
  _instance= None
  _lock= threading.Lock()
  def __new__(cls, *args, **kwargs):
    raise NotImplementedError('Do not initialize with the constructor.')
  @classmethod
  def new(cls, *args, **kwargs):
    if cls._instance is None:
      with cls._lock:
        if cls._instance is None:
          cls._instance= super(TDynamixelPortHandler,cls).__new__(cls)
          cls._instance.__init__(*args, **kwargs)
    return cls._instance

  def __init__(self):
    #Callback in the reopen thread.  If this function returns False, the thread ends.
    self.ReopenCallback= None

    #List of {'dev':DEVICE_NAME, 'port':PORT_HANDLER, 'baudrate':BAUD_RATE, 'locker':Thread locker, 'ref':Reference counter}
    self.opened= []
    #Reopen thread state:
    self.thread_reopen= [False,None]
    #List of callback functions called after reopening the ports.
    self.on_reopened= []

    #Time to sleep after closing a port.
    self.time_to_sleep_after_closing_port= 0.1

  #Set a callback function on_reopened: Callback function called after the port reopen.
  def SetOnReopened(self, on_reopened):
    if on_reopened is not None and on_reopened not in self.on_reopened:
      self.on_reopened.append(on_reopened)

  #Return port handler and the corresponding thread locker from device name.
  def Port(self, dev):
    try:
      opened= next(value for value in self.opened if value['dev']==dev)
      return opened['port'],opened['locker']
    except StopIteration:  return None,None

  #Open the device dev and set the baudrate if baudrate is not None.
  #If baudrate is None, reopen is True, and baudrate is saved in self.opened, it is set again.
  #Return the port number if succeeded, and None if failrue.
  #This will do nothing if the device is already opened (just returns the port number).
  def Open(self, dev='/dev/ttyUSB0', baudrate=None, reopen=False):
    set_baudrate= False
    try:
      #Check if dev is already opened.
      info= next(value for value in self.opened if value['dev']==dev)
      info['ref']+= 1  #Increase the reference counter.
      port_handler= info['port']
      if all((baudrate is None, reopen, info['baudrate']>0)):
        baudrate= info['baudrate']
        #info['baudrate']= 0
        set_baudrate= True

    except StopIteration:  #dev is not opened yet.
      info= {'port':None, 'dev':dev, 'baudrate':0, 'locker':threading.RLock(), 'ref':1}
      self.opened.append(info)

    with info['locker']:
      if info['port'] is None:
        # Initialize PortHandler Structs
        # Set the port path
        # Get methods and members of PortHandlerLinux or PortHandlerWindows
        port_handler= dynamixel.PortHandler(dev)

        #Open port
        if port_handler.openPort():
          print 'DxlPortHandler: Opened a port:', dev, port_handler
        else:
          print 'DxlPortHandler: Failed to open a port:', dev, port_handler
          port_handler.closePort()
          time.sleep(self.time_to_sleep_after_closing_port)
          return None
        info['port']= port_handler

      if (baudrate is not None and info['baudrate']!=baudrate) or set_baudrate:
        #Set port baudrate
        if port_handler.setBaudRate(int(baudrate)):
          print 'DxlPortHandler: Changed the baud rate:', dev, port_handler, baudrate
        else:
          print 'DxlPortHandler: Failed to change the baud rate to:', dev, port_handler, baudrate
          return None
        info['baudrate']= baudrate

    return port_handler

  #Close the port.  Specify the port by device name (dev) OR port handler (port).
  def Close(self, dev=None, port=None):
    i,info= -1,None
    port_handler= None
    if dev is not None:
      try:
        i,info= next((i,value) for i,value in enumerate(self.opened) if value['dev']==dev)
        port_handler= info['port']
      except StopIteration:
        pass
    elif port is not None:
      try:
        i,info= next((i,value) for i,value in enumerate(self.opened) if value['port']==port)
        port_handler= port
      except StopIteration:
        pass
    #else:
      #raise Exception('TDynamixelPortHandler.Close: dev or port must be specified.')
    if port_handler is not None and i>=0:
      self.opened[i]['ref']-= 1  #Decrease the reference counter.
      if self.opened[i]['ref']<=0:
        with self.opened[i]['locker']:
          port_handler.closePort()
          print 'DxlPortHandler: Closed the port:',info['dev'],port_handler
          time.sleep(self.time_to_sleep_after_closing_port)
          del self.opened[i]

  #Mark the device or port to be an error state.
  #Specify the port by device name (dev) OR port handler (port).
  def MarkError(self, dev=None, port=None):
    info= None
    port_handler= None
    if dev is not None:
      try:
        info= next(value for value in self.opened if value['dev']==dev)
        port_handler= info['port']
      except StopIteration:
        pass
    elif port is not None:
      try:
        info= next(value for value in self.opened if value['port']==port)
        port_handler= port
      except StopIteration:
        pass
    #else:
      #raise Exception('TDynamixelPortHandler.MarkError: dev or port must be specified.')
    if port_handler is not None:
      with info['locker']:
        port_handler.closePort()
        print 'DxlPortHandler.MarkError: Closed the port:',info['dev'],port_handler
        time.sleep(self.time_to_sleep_after_closing_port)
    if info is not None:  info['port']= None

  '''Start reopen thread to reopen all devices whose port is None.
    interval: Cycle of reopen attempt (sec). '''
  def StartReopen(self, interval=1.0):
    print 'DxlPortHandler: Reopen is requested'
    self.StopReopen()
    if len(self.opened)==0:
      print 'DxlPortHandler: Reopen canceled'
      return
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
      lockers= [value['locker'] for value in self.opened if value['port'] is None]
      for locker in lockers:  locker.acquire()
      try:
        success= True
        for info in self.opened:
          if info['port'] is not None:  continue
          port_handler= self.Open(dev=info['dev'], reopen=True)
          if port_handler is None:  success= False
        if success:
          for on_reopened in self.on_reopened:
            on_reopened()
      finally:
        for locker in lockers:  locker.release()
      time.sleep(interval)
    self.thread_reopen[0]= False

#Global object:
DxlPortHandler= TDynamixelPortHandler.new()


class TDynamixel1(object):
  def __init__(self, type, dev='/dev/ttyUSB0'):
    # For Dynamixel XM430-W350
    if type in ('XM430-W350','XH430-V350','XH540-W270','MX-64AR','XD540-T270'):
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
        'ACC_LIMIT'           : (40,4),  #'XH540-W270','XD540-T270' does not have this.
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
      if type in ('XH540-W270','XD540-T270'):
        self.ADDR['ACC_LIMIT']= (None,None)
      self.PROTOCOL_VERSION = 2  # Protocol version of Dynamixel

    # For Dynamixel PH54-200-S500-R, PM54-060-S250-R
    elif type in ('PH54-200-S500','PM54-060-S250'):
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
        'TORQUE_ENABLE'       : (512,1),
        'HARDWARE_ERR_ST'     : (518,1),
        'VEL_I_GAIN'          : (524,2),
        'VEL_P_GAIN'          : (526,2),
        'POS_D_GAIN'          : (528,2),
        'POS_I_GAIN'          : (530,2),
        'POS_P_GAIN'          : (532,2),
        'GOAL_PWM'            : (548,2),
        'GOAL_CURRENT'        : (550,2),
        'GOAL_VELOCITY'       : (552,4),
        'GOAL_POSITION'       : (564,4),
        'PRESENT_PWM'         : (572,2),
        'PRESENT_CURRENT'     : (574,2),
        'PRESENT_VELOCITY'    : (576,4),
        'PRESENT_POSITION'    : (580,4),
        'PRESENT_IN_VOLT'     : (592,2),
        'PRESENT_TEMP'        : (594,1),
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

    # For RH-P12-RN(A)
    elif type in ('RH-P12-RN(A)',):
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
        'TORQUE_ENABLE'       : (512,1),
        'HARDWARE_ERR_ST'     : (518,1),
        'VEL_I_GAIN'          : (524,2),
        'VEL_P_GAIN'          : (526,2),
        'POS_D_GAIN'          : (528,2),
        'POS_I_GAIN'          : (530,2),
        'POS_P_GAIN'          : (532,2),
        'GOAL_PWM'            : (548,2),
        'GOAL_CURRENT'        : (550,2),
        'GOAL_VELOCITY'       : (552,4),
        'GOAL_POSITION'       : (564,4),
        'PRESENT_PWM'         : (572,2),
        'PRESENT_CURRENT'     : (574,2),
        'PRESENT_VELOCITY'    : (576,4),
        'PRESENT_POSITION'    : (580,4),
        'PRESENT_IN_VOLT'     : (592,2),
        'PRESENT_TEMP'        : (594,1),
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
    #NOTE: 'RH-P12-RN','RH-P12-RN(A)': CURRENT and CURRPOS are available.
    #NOTE: 'PH54-200-S500','PM54-060-S250': CURRENT, VELOCITY, POSITION(default), EXTPOS, PWM are available.

    # Baud rates                                  0     1      2   3   4   5   6     7      8      9
    self.BAUD_RATE=                           [9600,57600,115200,1e6,2e6,3e6,4e6,4.5e6]
    if type=='RH-P12-RN':     self.BAUD_RATE= [9600,57600,115200,1e6,2e6,3e6,4e6,4.5e6,10.5e6]
    if type=='RH-P12-RN(A)':  self.BAUD_RATE= [9600,57600,115200,1e6,2e6,3e6,4e6,4.5e6,   6e6,10.5e6]
    if type=='PH54-200-S500': self.BAUD_RATE= [9600,57600,115200,1e6,2e6,3e6,4e6,4.5e6,   6e6,10.5e6]
    if type=='PM54-060-S250': self.BAUD_RATE= [9600,57600,115200,1e6,2e6,3e6,4e6,4.5e6,   6e6,10.5e6]

    self.TORQUE_ENABLE  = 1  # Value for enabling the torque
    self.TORQUE_DISABLE = 0  # Value for disabling the torque
    self.MIN_POSITION = 0  # Dynamixel will rotate between this value
    self.MAX_POSITION = 4095  # and this value (note that the Dynamixel would not move when the position value is out of movable range. Check e-manual about the range of the Dynamixel you use.)
    if type in ('RH-P12-RN','RH-P12-RN(A)'):
      self.MAX_POSITION = 1150
    elif type=='PH54-200-S500':
      self.MIN_POSITION = -501433
      self.MAX_POSITION = 501433
    elif type=='PM54-060-S250':
      self.MIN_POSITION = -251173
      self.MAX_POSITION = 251173
    if type=='XM430-W350':    self.MAX_CURRENT = 1193   # == Current Limit(38)
    elif type=='XH430-V350':  self.MAX_CURRENT = 689   # == Current Limit(38)
    elif type in ('XH540-W270','XD540-T270'):  self.MAX_CURRENT = 2047   # == Current Limit(38)
    elif type=='MX-64AR':     self.MAX_CURRENT = 1941   # == Current Limit(38)
    elif type=='RH-P12-RN':   self.MAX_CURRENT = 820
    elif type=='RH-P12-RN(A)':   self.MAX_CURRENT = 1984
    elif type=='PH54-200-S500':   self.MAX_CURRENT = 22740
    elif type=='PM54-060-S250':   self.MAX_CURRENT = 7980
    self.MAX_PWM = 885
    if type=='RH-P12-RN':  self.MAX_PWM = None
    elif type=='RH-P12-RN(A)':  self.MAX_PWM = 2009
    elif type=='PH54-200-S500':  self.MAX_PWM = 2009
    elif type=='PM54-060-S250':  self.MAX_PWM = 2009

    self.POSITION_UNIT= math.pi/2048.0
    self.POSITION_OFFSET= 2048.0
    if type=='PH54-200-S500':
      self.POSITION_UNIT= math.pi/501923.0
      self.POSITION_OFFSET= 0.0
    elif type=='PM54-060-S250':
      self.POSITION_UNIT= math.pi/251417.0
      self.POSITION_OFFSET= 0.0
    if type=='XM430-W350':
      self.CURRENT_UNIT= 2.69
      self.VELOCITY_UNIT= 0.229*(2.0*math.pi)/60.0
    elif type=='XH430-V350':
      self.CURRENT_UNIT= 1.34
      self.VELOCITY_UNIT= 0.229*(2.0*math.pi)/60.0
    elif type in ('XH540-W270','XD540-T270'):
      self.CURRENT_UNIT= 2.69
      self.VELOCITY_UNIT= 0.229*(2.0*math.pi)/60.0
    elif type=='MX-64AR':
      self.CURRENT_UNIT= 3.36
      self.VELOCITY_UNIT= 0.229*(2.0*math.pi)/60.0
    elif type=='RH-P12-RN':
      self.CURRENT_UNIT= 4.02
      self.VELOCITY_UNIT= 0.114*(2.0*math.pi)/60.0
    elif type=='RH-P12-RN(A)':
      self.CURRENT_UNIT= 1.0
      self.VELOCITY_UNIT= 0.01*(2.0*math.pi)/60.0
    elif type=='PH54-200-S500':
      self.CURRENT_UNIT= 1.0
      self.VELOCITY_UNIT= 0.01*(2.0*math.pi)/60.0
    elif type=='PM54-060-S250':
      self.CURRENT_UNIT= 1.0
      self.VELOCITY_UNIT= 0.01*(2.0*math.pi)/60.0

    # Configurable variables
    self.Id= 1  # Dynamixel ID
    self.Baudrate= 57600
    #self.Baudrate= 1000000
    self.DevName= dev  # Port name.  e.g. Win:'COM1', Linux:'/dev/ttyUSB0'
    self.OpMode= 'POSITION'
    if type in ('RH-P12-RN','RH-P12-RN(A)'):  self.OpMode= 'CURRPOS'
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
    self.port_handler= lambda self=self:DxlPortHandler.Port(self.DevName)
    self.packet_handler= None
    self.dxl_result= None
    self.dxl_err= None

    #Dictionary to memorize the current state.
    self.state_memory= {}

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
    port_handler,port_locker= self.port_handler()
    if port_handler is None:
      print 'Port {dev} is closed.'.format(dev=self.DevName)
      return
    addr,size= self.ADDR[address]
    if addr is None:
      print '{address} is not available with this Dynamixel.'.format(address=address)
      return
    with port_locker:
      self.dxl_result,self.dxl_err= self.WriteFuncs[size](port_handler, self.Id, addr, value)
    self.state_memory[address]= value

  def Read(self, address):
    port_handler,port_locker= self.port_handler()
    if port_handler is None:
      print 'Port {dev} is closed.'.format(dev=self.DevName)
      return None
    addr,size= self.ADDR[address]
    if addr is None:
      print '{address} is not available with this Dynamixel.'.format(address=address)
      return None
    with port_locker:
      value,self.dxl_result,self.dxl_err= self.ReadFuncs[size](port_handler, self.Id, addr)
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
    if size==4:
      value= value & 4294967295
      if value>2147483647:  value= -(4294967296-value)
    return value

  def Setup(self):
    DxlPortHandler.Open(dev=self.DevName, baudrate=self.Baudrate)
    if self.port_handler()[0] is None:
      return False

    # Initialize PacketHandler
    self.packet_handler= dynamixel.PacketHandler(self.PROTOCOL_VERSION)
    self.WriteFuncs= (None, self.packet_handler.write1ByteTxRx, self.packet_handler.write2ByteTxRx, None, self.packet_handler.write4ByteTxRx)
    self.ReadFuncs= (None, self.packet_handler.read1ByteTxRx, self.packet_handler.read2ByteTxRx, None, self.packet_handler.read4ByteTxRx)

    self.SetOpMode(self.OP_MODE[self.OpMode])
    #self.EnableTorque()

    self.MemorizeState()
    DxlPortHandler.SetOnReopened(self.RecallState)
    return True

  def Quit(self):
    #self.DisableTorque()
    # Close port
    DxlPortHandler.Close(dev=self.DevName)

  #Memorize the current RAM state for setting them again after reopen.
  def MemorizeState(self):
    for address,(addr,size) in self.ADDR.iteritems():
      if addr is None:  continue
      self.state_memory[address]= self.Read(address)

  def RecallState(self):
    port_handler,port_locker= self.port_handler()
    if port_handler is None:
      print 'TDynamixel1.RecallState failed as port {dev} is still closed.'.format(dev=self.DevName)
      return
    with port_locker:
      recall= ('TORQUE_ENABLE','VEL_I_GAIN','VEL_P_GAIN','POS_D_GAIN','POS_I_GAIN','POS_P_GAIN',)
      for address in recall:
        if address in self.state_memory:
          print 'RecallState: {address}, {value}'.format(address=address,value=self.state_memory[address])
          self.Write(address,self.state_memory[address])

  #Check the result of sending a command.
  #Print the error message if quiet is False.
  #If reopen is True, we start the port reopen thread.
  def CheckTxRxResult(self, quiet=False, reopen=True):
    normal= self.dxl_result==dynamixel.COMM_SUCCESS and self.dxl_err==0
    if not normal:# and not quiet:
      print 'dxl_result=',self.dxl_result, 'dxl_err=',self.dxl_err
      if self.dxl_result != dynamixel.COMM_SUCCESS:
        print self.packet_handler.getTxRxResult(self.dxl_result)
      if self.dxl_err != 0:
        print self.packet_handler.getRxPacketError(self.dxl_err)
    if not normal:
      DxlPortHandler.MarkError(dev=self.DevName)
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
    for address,(addr,size) in self.ADDR.iteritems():
      if addr is None:  continue
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
    port_handler,port_locker= self.port_handler()
    if port_handler is None:
      print 'TDynamixel1.Reboot: Port {dev} is closed.'.format(dev=self.DevName)
    with port_locker:
      self.dxl_result,self.dxl_err= self.packet_handler.reboot(port_handler, self.Id)
    self.CheckTxRxResult()

  #Factory-reset Dynamixel
  #mode: 0xFF : reset all values (ID to 1, baudrate to 57600).
  #      0x01 : reset all values except ID (baudrate to 57600).
  #      0x02 : reset all values except ID and baudrate.
  def FactoryReset(self, mode=0x02):
    port_handler,port_locker= self.port_handler()
    if port_handler is None:
      print 'TDynamixel1.FactoryReset: Port {dev} is closed.'.format(dev=self.DevName)
    with port_locker:
      self.dxl_result,self.dxl_err= self.packet_handler.factoryReset(port_handler, self.Id, mode)
    self.CheckTxRxResult()

  #Move the position to a given value.
  #  target: Target position, should be in [self.MIN_POSITION, self.MAX_POSITION]
  #  blocking: True: this function waits the target position is reached.  False: this function returns immediately.
  def MoveTo(self, target, blocking=True):
    target = int(target)
    #FIXME: If OpMode allows multi turn, target could vary.
    if self.OpMode!='EXTPOS':
      if target < self.MIN_POSITION:  target = self.MIN_POSITION
      elif target > self.MAX_POSITION:  target = self.MAX_POSITION

    # Write goal position
    self.Write('GOAL_POSITION', target)
    #if not self.CheckTxRxResult():  return
    self.CheckTxRxResult()
    #print 'debug:TDynamixel1:MoveTo:',target

    p_log= []  #For detecting stuck.
    while blocking:
      pos= self.Position()
      if pos is None:  return
      p_log.append(pos)
      if len(p_log)>50:  p_log.pop(0)
      if not (abs(target - p_log[-1]) > self.GoalThreshold):  break
      #Detecting stuck:
      if len(p_log)>=50 and not (abs(p_log[0] - p_log[-1]) > self.GoalThreshold):
        print 'TDynamixel1:MoveTo: Control gets stuck. Abort.'
        break

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

    p_log= []  #For detecting stuck.
    while blocking:
      pos= self.Position()
      if pos is None:  return
      p_log.append(pos)
      if len(p_log)>50:  p_log.pop(0)
      if not (abs(target - p_log[-1]) > self.GoalThreshold):  break
      #Detecting stuck:
      if len(p_log)>=50 and not (abs(p_log[0] - p_log[-1]) > self.GoalThreshold):
        print 'TDynamixel1:MoveToC: Control gets stuck. Abort.'
        break

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


