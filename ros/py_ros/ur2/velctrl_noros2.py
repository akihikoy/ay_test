#!/usr/bin/python3
#\file    velctrl_noros2.py
#\brief   Velocity control with direct connection to UR over Ethernet (no ROS).
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jun.27, 2018
import socket
import struct

class TURVersion(object):
  def __init__(self):
    self.timestamp= 0  #uint64_t
    self.source= 0  #int8_t
    self.robot_message_type= 0  #int8_t
    self.project_name_size= 0  #int8_t
    self.project_name= ''  #char[15]
    self.major_version= 0  #int8_t
    self.minor_version= 0  #int8_t
    self.svn_revision= 0  #int
    self.build_date= ''  #char[25]

class TURState(object):
  ROBOT_STATE= 16
  ROBOT_MESSAGE= 20
  PROGRAM_STATE_MESSAGE= 25
  ROBOT_MESSAGE_TEXT= 0
  ROBOT_MESSAGE_PROGRAM_LABEL= 1
  PROGRAM_STATE_MESSAGE_VARIABLE_UPDATE= 2
  ROBOT_MESSAGE_VERSION= 3
  ROBOT_MESSAGE_SAFETY_MODE= 5
  ROBOT_MESSAGE_ERROR_CODE= 6
  ROBOT_MESSAGE_KEY= 7
  ROBOT_MESSAGE_REQUEST_VALUE= 9
  ROBOT_MESSAGE_RUNTIME_EXCEPTION= 10

  def __init__(self):
    self.version= TURVersion()

  def GetVersion(self):
    return self.version.major_version + 0.1*self.version.minor_version + 0.0000001*self.version.svn_revision

  def Unpack(self, buf):
    offset= 0
    while len(buf)>offset:
      l,message_type= struct.unpack_from("!IB", buf, offset)
      if l+offset>len(buf):  return
      if message_type==self.ROBOT_MESSAGE:
        self.UnpackRobotMessage(buf, offset, l)
      elif message_type==self.ROBOT_STATE:
        #self.UnpackRobotState(buf, offset, l)
        pass
      if message_type==self.PROGRAM_STATE_MESSAGE:
        pass
      offset+= l

  def UnpackRobotMessage(self, buf, offset, l):
    offset+= 5
    timestamp,source,robot_message_type= struct.unpack_from("!Qbb", buf, offset)
    offset+= 8+1+1
    if robot_message_type==self.ROBOT_MESSAGE_VERSION:
      self.version.timestamp= timestamp
      self.version.source= source
      self.version.robot_message_type= robot_message_type
      self.version.project_name_size= struct.unpack_from("!b", buf, offset)[0]
      offset+= 1
      self.version.project_name= buf[offset:offset+self.version.project_name_size]
      offset+= self.version.project_name_size
      mj,mi,svn= struct.unpack_from("!bbI", buf, offset)
      self.version.major_version,self.version.minor_version,self.version.svn_revision= mj,mi,svn
      offset+= 1+1+4
      self.version.build_date= buf[offset:l]
      #if version_msg_.major_version<2:
        #robot_mode_running_= ROBOT_RUNNING_MODE

def GetURState(robot_hostname):
  port= 30001
  socketobj= socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  socketobj.connect((robot_hostname, port))

  robot_state= TURState()
  buf= socketobj.recv(512)
  socketobj.close()
  robot_state.Unpack(buf)
  return robot_state

#Velocity control interface of UR.
class TURVelCtrl(object):
  def __init__(self, robot_hostname):
    self.robot_hostname= robot_hostname

  def Init(self):
    self.robot_state= GetURState(self.robot_hostname)
    self.robot_ver= self.robot_state.GetVersion()
    port= 30003
    #socketobj= socket.create_connection((robot_hostname, port))
    self.socketobj= socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self.socketobj.connect((self.robot_hostname, port))

  def Step(self, dq, acc):
    #This swich for the version was implemented previously as the handler of the joint_speed topic,
    #but as of 20190529 that topic is removed and the alternative hardware interface uses the following.
    #if self.robot_ver>=3.3:
      #cmd= "speedj([%1.5f, %1.5f, %1.5f, %1.5f, %1.5f, %1.5f], %f, 0.008)\n" % (
              #dq[0],dq[1],dq[2],dq[3],dq[4],dq[5],acc)
    #elif self.robot_ver>=3.1:
      #cmd= "speedj([%1.5f, %1.5f, %1.5f, %1.5f, %1.5f, %1.5f], %f)\n" % (
              #dq[0],dq[1],dq[2],dq[3],dq[4],dq[5],acc)
    #else:
      #cmd= "speedj([%1.5f, %1.5f, %1.5f, %1.5f, %1.5f, %1.5f], %f, 0.02)\n" % (
              #dq[0],dq[1],dq[2],dq[3],dq[4],dq[5],acc)
    cmd= "speedj([%1.5f, %1.5f, %1.5f, %1.5f, %1.5f, %1.5f], %f, 0.02)\n" % (
            dq[0],dq[1],dq[2],dq[3],dq[4],dq[5],acc)
    self.AddCommandToQueue(cmd)

  def Finish(self):
    self.Step([0.0]*6, 10.0)
    self.socketobj.close()

  def AddCommandToQueue(self, cmd):
    if cmd[-1]!='\n': cmd+= '\n'
    len_sent= self.socketobj.send(cmd)
    if len_sent!=len(cmd):
      raise Exception('Command was not sent properly; cmd {cmd} byte, sent {sent} byte'.format(cmd=len(cmd),sent=len_sent))


if __name__=='__main__':
  import math,time,sys
  from rate_adjust import TRate

  #robot_hostname,is_e= 'ur3a',False
  robot_hostname,is_e= 'ur3ea',True
  print('robot & version:',robot_hostname,GetURState(robot_hostname).GetVersion())
  res= input('continue? > ')
  if len(res)>0 and res[0] not in ('y','Y',' '):  sys.exit(0)

  velctrl= TURVelCtrl(robot_hostname)
  velctrl.Init()

  t0= time.time()
  if not is_e:  rate= TRate(125)  #UR receives velocities at 125 Hz.
  else:  rate= TRate(500)  #UR receives velocities at 500 Hz.

  try:
    while True:
      #dq= [0.0]*6
      t= time.time()-t0
      dq= [0.08*math.sin(math.pi*t)]*6
      acc= 10.0
      velctrl.Step(dq, acc)
      rate.sleep()

  except KeyboardInterrupt:
    print('Interrupted')

  finally:
    velctrl.Finish()
