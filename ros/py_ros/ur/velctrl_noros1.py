#!/usr/bin/python3
#\file    velctrl_noros1.py
#\brief   Velocity control with direct connection to UR over Ethernet (no ROS).
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jun.27, 2018
import socket
import struct
from rate_adjust import TRate

def AddCommandToQueue(socketobj, cmd):
  if cmd[-1]!='\n': cmd+= '\n'
  len_sent= socketobj.send(cmd)
  if len_sent!=len(cmd):
    print('Command was not sent properly; cmd {cmd} byte, sent {sent} byte'.format(cmd=len(cmd),sent=len_sent))

class TRobotState(object):
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

  class TVersion(object):
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

  def __init__(self):
    self.version= TRobotState.TVersion()

  def GetVersion(self):
    return self.version.major_version + 0.1*self.version.minor_version + 0.0000001*self.version.svn_revision

  def Unpack(self, buf):
    offset= 0
    while len(buf)>offset:
      #int len;
      #unsigned char message_type;
      #memcpy(&len, &buf[offset], sizeof(len));
      l= struct.unpack_from("!I", buf, offset)[0]
      if l+offset>len(buf):  return
      #memcpy(&message_type, &buf[offset + sizeof(len)], sizeof(message_type));
      message_type= struct.unpack_from("!B", buf, offset+4)[0]
      #print len(buf), l, message_type, message_type==self.ROBOT_MESSAGE
      if message_type==self.ROBOT_MESSAGE:
        #RobotState::unpackRobotMessage(buf, offset, len); //'len' is inclusive the 5 bytes from messageSize and messageType
        self.UnpackRobotMessage(buf, offset, l)
      elif message_type==self.ROBOT_STATE:
        #RobotState::unpackRobotState(buf, offset, len); //'len' is inclusive the 5 bytes from messageSize and messageType
        #self.UnpackRobotState(buf, offset, l)
        pass
      if message_type==self.PROGRAM_STATE_MESSAGE:
        pass
      offset+= l

  def UnpackRobotMessage(self, buf, offset, l):
    offset+= 5
    #uint64_t timestamp;
    #int8_t source, robot_message_type;
    #memcpy(&timestamp, &buf[offset], sizeof(timestamp));
    timestamp= struct.unpack_from("!Q", buf, offset)[0]
    offset+= 8
    #memcpy(&source, &buf[offset], sizeof(source));
    source= struct.unpack_from("!b", buf, offset)[0]
    offset+= 1
    #memcpy(&robot_message_type, &buf[offset], sizeof(robot_message_type));
    robot_message_type= struct.unpack_from("!b", buf, offset)[0]
    offset+= 1
    #print 'robot_message_type',robot_message_type, self.ROBOT_MESSAGE_VERSION, robot_message_type==self.ROBOT_MESSAGE_VERSION
    if robot_message_type==self.ROBOT_MESSAGE_VERSION:
      self.version.timestamp= timestamp
      self.version.source= source
      self.version.robot_message_type= robot_message_type
      #self.UnpackRobotMessageVersion(buf, offset, l)
      self.version.project_name_size= struct.unpack_from("!b", buf, offset)[0]
      offset+= 1
      self.version.project_name= buf[offset:offset+self.version.project_name_size]
      offset+= self.version.project_name_size
      self.version.major_version= struct.unpack_from("!b", buf, offset)[0]
      offset+= 1
      self.version.minor_version= struct.unpack_from("!b", buf, offset)[0]
      offset+= 1
      self.version.svn_revision= struct.unpack_from("!I", buf, offset)[0]
      offset+= 4
      self.version.build_date= buf[offset:l]
      #print self.version.__dict__
      #if version_msg_.major_version<2:
        #robot_mode_running_= ROBOT_RUNNING_MODE

  def UnpackRobotMessageVersion(self, buf, offset, l):
    #uint64_t timestamp;
    #int8_t source;
    #int8_t robot_message_type;
    #int8_t project_name_size;
    #char project_name[15];
    #uint8_t major_version;
    #uint8_t minor_version;
    #int svn_revision;
    #char build_date[25];

    #memcpy(&version_msg_.project_name_size, &buf[offset],
            #sizeof(version_msg_.project_name_size));
    #offset += sizeof(version_msg_.project_name_size);
    project_name_size= struct.unpack_from("!b", buf, offset)[0]
    offset+= 1
    #memcpy(&version_msg_.project_name, &buf[offset],
            #sizeof(char) * version_msg_.project_name_size);
    #offset += version_msg_.project_name_size;
    #version_msg_.project_name[version_msg_.project_name_size] = '\0';
    project_name= buf[offset:offset+project_name_size]
    offset+= project_name_size
    #memcpy(&version_msg_.major_version, &buf[offset],
            #sizeof(version_msg_.major_version));
    #offset += sizeof(version_msg_.major_version);
    major_version= struct.unpack_from("!b", buf, offset)[0]
    offset+= 1
    #memcpy(&version_msg_.minor_version, &buf[offset],
            #sizeof(version_msg_.minor_version));
    #offset += sizeof(version_msg_.minor_version);
    minor_version= struct.unpack_from("!b", buf, offset)[0]
    offset+= 1
    #memcpy(&version_msg_.svn_revision, &buf[offset],
            #sizeof(version_msg_.svn_revision));
    #offset += sizeof(version_msg_.svn_revision);
    svn_revision= struct.unpack_from("!I", buf, offset)[0]
    offset+= 1
    #memcpy(&version_msg_.build_date, &buf[offset], sizeof(char) * len - offset);
    build_date= buf[offset:l-offset]
    #if version_msg_.major_version<2:
      #robot_mode_running_= ROBOT_RUNNING_MODE

def GetRobotState(robot_hostname):
  port= 30001
  socketobj= socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  socketobj.connect((robot_hostname, port))

  robot_state= TRobotState()
  buf= socketobj.recv(512)
  #print buf
  robot_state.Unpack(buf)

  socketobj.close()

  print(robot_state.version.major_version, robot_state.version.minor_version)
  print(robot_state.GetVersion())
  return robot_state

if __name__=='__main__':
  robot_hostname= 'ur3a'

  robot_state= GetRobotState(robot_hostname)

  port= 30003
  #socketobj= socket.create_connection((robot_hostname, port))
  socketobj= socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  socketobj.connect((robot_hostname, port))

  #robot_state= TRobotState()
  #buf= socketobj.recv(512)
  ##print buf
  #robot_state.Unpack(buf)
  #print robot_state.version.major_version, robot_state.version.minor_version

  rate= TRate(125)  #UR receives velocities at 125 Hz.

  try:
    while True:
      vel= [0.0]*6
      acc= 10.0
      ver= robot_state.GetVersion()
      if ver>=3.3:
        cmd= "speedj([%1.5f, %1.5f, %1.5f, %1.5f, %1.5f, %1.5f], %f, 0.008)\n" % (
                vel[0],vel[1],vel[2],vel[3],vel[4],vel[5],acc)
      elif ver>=3.1:
        cmd= "speedj([%1.5f, %1.5f, %1.5f, %1.5f, %1.5f, %1.5f], %f)\n" % (
                vel[0],vel[1],vel[2],vel[3],vel[4],vel[5],acc)
      else:
        cmd= "speedj([%1.5f, %1.5f, %1.5f, %1.5f, %1.5f, %1.5f], %f, 0.02)\n" % (
                vel[0],vel[1],vel[2],vel[3],vel[4],vel[5],acc)
      AddCommandToQueue(socketobj,cmd)

      rate.sleep()

  except KeyboardInterrupt:
    print('Interrupted')

  finally:
    socketobj.close()
