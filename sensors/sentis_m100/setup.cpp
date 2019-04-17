//-------------------------------------------------------------------------------------------
/*! \file    setup.cpp
    \brief   Setup program for sentis m100
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    Oct.22, 2014

    Compile:
    g++ -g -Wall -rdynamic -O2 setup.cpp -o setup -I/usr/include/libm100 -lm100
*/
//-------------------------------------------------------------------------------------------
#include <m100api.h>
#include <iostream>
#include <iomanip>
//-------------------------------------------------------------------------------------------
/* Modify the definition of XYZ_COORDS_DATA, which is originally defined in
  libm100/apitypes.h as 0x0010 but according to Sentis-ToF-M100_UM_V1.pdf
  Sec 8.6.1 (General registers - ImageDataFormat), it should be 0x0018.
  So, this seems to be a bug.  */
#undef XYZ_COORDS_DATA
#define XYZ_COORDS_DATA 0x0018
//-------------------------------------------------------------------------------------------
#define M100_IMAGE_WIDTH 160
#define M100_IMAGE_HEIGHT 120
#define M100_IMAGE_SIZE (M100_IMAGE_WIDTH*M100_IMAGE_HEIGHT)
//-------------------------------------------------------------------------------------------
// m100 registers.  See Sentis-ToF-M100_UM_V1.pdf
#define ModulationFrequency     0x0009
#define LedboardTemp      0x001B
#define CalibrationCommand      0x000F
#define CalibrationExtended     0x0021
#define FrameTime         0x001F
#define TempCompGradient2       0x0030
#define TempCompGradient3       0x003C
#define BuildYearMonth    0x003D
#define BuildDayHour      0x003E
#define BuildMinuteSecond       0x003F
#define UpTimeLow         0x0040
#define UpTimeHigh        0x0041
#define AkfPlausibilityCheckAmpLimit  0x0042
#define CommKeepAliveTimeout    0x004E
#define CommKeepAliveReset      0x004F
#define AecAvgWeight0     0x01A9
#define AecAvgWeight1     0x01AA
#define AecAvgWeight2     0x01AB
#define AecAvgWeight3     0x01AC
#define AecAvgWeight4     0x01AD
#define AecAvgWeight5     0x01AE
#define AecAvgWeight6     0x01AF
#define AecAmpTarget      0x01B0
#define AecTintStepMax    0x01B1
#define AecTintMax        0x01B2
#define AecKp       0x01B3
#define AecKi       0x01B4
#define AecKd       0x01B5
//-------------------------------------------------------------------------------------------
// m100 registers.  See Sentis-ToF-M100_UM_V1.pdf
#define CmdExecPassword     0x0022
//-------------------------------------------------------------------------------------------
// m100 registers.  See Sentis-ToF-M100_UM_V1.pdf
#define Eth0Config    0x0240
#define Eth0Mac2      0x0241
#define Eth0Mac1      0x0242
#define Eth0Mac0      0x0243
#define Eth0Ip0       0x0244
#define Eth0Ip1       0x0245
#define Eth0Snm0      0x0246
#define Eth0Snm1      0x0247
#define Eth0Gateway0        0x0248
#define Eth0Gateway1        0x0249
#define Eth0TcpStreamPort   0x024A
#define Eth0TcpConfigPort   0x024B
#define Eth0UdpStreamIp0    0x024C
#define Eth0UdpStreamIp1    0x024D
#define Eth0UdpStreamPort   0x024E
//-------------------------------------------------------------------------------------------
namespace loco_rabbits
{

//===========================================================================================
class TSentisM100
//===========================================================================================
{
public:
  #define HEX2(x_val) std::hex<<std::setfill('0')<<std::setw(2)<<(x_val)<<std::dec
  #define HEX4(x_val) std::hex<<std::setfill('0')<<std::setw(4)<<(x_val)<<std::dec

  TSentisM100() : handle_(NULL), error_(0), data_format_(DEPTH_AMP_DATA) {}

  /*! Establish the connection and initialize the sensor.
    \param [in]init_fps     Initial FPS (1-40 Hz).
    \param [in]data_format  Data format.  Choose from {DEPTH_AMP_DATA, XYZ_COORDS_DATA, XYZ_AMP_DATA}.
  */
  bool Init(
      unsigned short init_fps= 1,
      unsigned short data_format= DEPTH_AMP_DATA,
      const char *tcp_ip= "192.168.0.10",
      const char *udp_ip= "224.0.0.1",
      unsigned short tcp_port= 10001,
      unsigned short udp_port= 10002);

  bool SetDHCP(bool using_dhcp);

  //! Set the frame rate (1-40 Hz; 45 Hz seems to work)
  bool SetFrameRate(unsigned short frame_rate);

  //! Set the frame rate to 1 Hz (lowest FPS) to cool down
  bool Sleep()  {return SetFrameRate(1);}

  //! Save RegMap to flash
  bool SaveToFlash();

  bool GetData();

  //! Print a part of registers.
  bool PrintRegisters(int preset);

  bool IsNoError(const char *msg=NULL)
    {
      if(error_==0)  return true;
      if(msg!=NULL)  std::cerr<<msg<<"Error: "<<error_<<std::endl;
      return false;
    }

  int CalcFrameSize()
    {
      switch(data_format_)
      {
      case DEPTH_AMP_DATA:   return 2*M100_IMAGE_SIZE*sizeof(unsigned short);
      case XYZ_COORDS_DATA:  return 3*M100_IMAGE_SIZE*sizeof(unsigned short);
      case XYZ_AMP_DATA:     return 4*M100_IMAGE_SIZE*sizeof(unsigned short);
      }
      return -1;
    }

  unsigned short ReadRegister(unsigned short address)
    {
      unsigned short res(0);
      error_= STSreadRegister(handle_, address, &res, 0, 0);
      return res;
    }
  bool WriteRegister(unsigned short address, unsigned short value)
    {
      error_= STSwriteRegister(handle_, address, value);
      return IsNoError();
    }
  bool PrintRegister(unsigned short address, const char *name="")
    {
      unsigned short res= ReadRegister(address);
      if(!IsNoError())  return false;
      std::cerr<<name<<": 0x"<<HEX4(res)
          <<", "<<res
          <<std::endl;
      return true;
    }
  bool PrintIPAddrRegister(unsigned short addr0, unsigned short addr1, const char *name="")
    {
      unsigned short res1= ReadRegister(addr1);
      if(!IsNoError())  return false;
      unsigned short res0= ReadRegister(addr0);
      if(!IsNoError())  return false;
      std::cerr<<name<<": "
          <<((res1&0xFF00)>>8)<<"."<<(res1&0x00FF)
          <<"."<<((res0&0xFF00)>>8)<<"."<<(res0&0x00FF)
          <<std::endl;
      return true;
    }
  bool PrintMacAddrRegister(unsigned short addr0, unsigned short addr1, unsigned short addr2, const char *name="")
    {
      unsigned short res2= ReadRegister(addr2);
      if(!IsNoError())  return false;
      unsigned short res1= ReadRegister(addr1);
      if(!IsNoError())  return false;
      unsigned short res0= ReadRegister(addr0);
      if(!IsNoError())  return false;
      std::cerr<<name<<": "
          <<HEX2((res2&0xFF00)>>8)<<":"<<HEX2(res2&0x00FF)
          <<":"<<HEX2((res1&0xFF00)>>8)<<":"<<HEX2(res1&0x00FF)
          <<":"<<HEX2((res0&0xFF00)>>8)<<":"<<HEX2(res0&0x00FF)
          <<std::dec<<std::endl;
      return true;
    }
public:
  // Accessors
  T_SENTIS_HANDLE& Handle() {return handle_;}
  const T_ERROR_CODE& ErrorCode() const {return error_;}
  unsigned short DataFormat() const {return data_format_;}
  const T_SENTIS_DATA_HEADER& DataHeader() const {return header_;}
  int FrameSize() const {return frame_size_/sizeof(unsigned short);}
  const unsigned short *const Buffer() const {return buffer_;}

private:
  T_SENTIS_HANDLE handle_;
  T_ERROR_CODE error_;
  unsigned short data_format_;

  T_SENTIS_DATA_HEADER header_;
  int frame_size_;
  /*! Buffer to store the observed data.  Its side depends on the data format.
      See Sentis-ToF-M100_UM_V1.pdf Sec 6.3 Camera Data Format.
      DEPTH_AMP_DATA: 2*M100_IMAGE_SIZE,
      XYZ_COORDS_DATA: 3*M100_IMAGE_SIZE,
      XYZ_AMP_DATA: 4*M100_IMAGE_SIZE. */
  unsigned short buffer_[4*M100_IMAGE_SIZE];

  #undef HEX2
  #undef HEX4
};
//-------------------------------------------------------------------------------------------

}
//-------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------
namespace loco_rabbits
{

//===========================================================================================
// class TSentisM100
//===========================================================================================

/*! Establish the connection and initialize the sensor.
  \param [in]init_fps     Initial FPS (1-40 Hz).
  \param [in]data_format  Data format.  Choose from {DEPTH_AMP_DATA, XYZ_COORDS_DATA, XYZ_AMP_DATA}.
*/
bool TSentisM100::Init(
      unsigned short init_fps,
      unsigned short data_format,
      const char *tcp_ip,
      const char *udp_ip,
      unsigned short tcp_port,
      unsigned short udp_port)
{
  T_SENTIS_CONFIG config;
  config.tcp_ip= tcp_ip;
  config.udp_ip= udp_ip;
  config.tcp_port= tcp_port;
  config.udp_port= udp_port;
  config.flags= HOLD_CONTROL_ALIVE;

  data_format_= data_format;

  // Connect with the device
  std::cerr<<"Connecting to "<<tcp_ip<<std::endl;
  handle_= STSopen(&config, &error_);
  if(!IsNoError("Connection failed. "))  return false;

  if(!PrintRegister(Mode1,"Mode1"))  return false;
  std::cerr<<"Setting auto exposure."<<std::endl;
  if(!WriteRegister(Mode1, (ReadRegister(Mode1) | 0x8)))  return false;
  if(!PrintRegister(Mode1,"Mode1"))  return false;

  if(!PrintRegister(FrameRate,"FrameRate"))  return false;
  std::cerr<<"Setting frame rate."<<std::endl;
  if(!WriteRegister(FrameRate, init_fps))  return false;  // up to 40
  if(!PrintRegister(FrameRate,"FrameRate"))  return false;

  if(!PrintRegister(ImageDataFormat,"ImageDataFormat"))  return false;
  std::cerr<<"Setting image data format."<<std::endl;
  if(!WriteRegister(ImageDataFormat, data_format_))  return false;
  if(!PrintRegister(ImageDataFormat,"ImageDataFormat"))  return false;

  return true;
}
//-------------------------------------------------------------------------------------------

bool TSentisM100::SetDHCP(bool using_dhcp)
{
  if(!PrintRegister(Eth0Config,"Eth0Config"))  return false;
  std::cerr<<"Setting DHCP mode."<<std::endl;
  unsigned short new_config= ReadRegister(Eth0Config);
  if(using_dhcp)  new_config= (new_config|0x0001);
  else            new_config= (new_config&0xFFFE);
  if(!WriteRegister(Eth0Config, new_config))  return false;
  if(!PrintRegister(Eth0Config,"Eth0Config"))  return false;
  std::cerr<<"  Note: in order to activate the DHCP mode, "
      "it is better to execute SaveToFlash, then restart the sensor."<<std::endl;
  return true;
}
//-------------------------------------------------------------------------------------------

//! Set the frame rate (1-40 Hz)
bool TSentisM100::SetFrameRate(unsigned short frame_rate)
{
  if(!PrintRegister(FrameRate,"FrameRate"))  return false;
  std::cerr<<"Setting frame rate."<<std::endl;
  if(!WriteRegister(FrameRate, frame_rate))  return false;
  if(!PrintRegister(FrameRate,"FrameRate"))  return false;
  return true;
}
//-------------------------------------------------------------------------------------------

bool TSentisM100::GetData()
{
  frame_size_= CalcFrameSize();
  // std::cerr<<"frame_size_:"<<frame_size_<<std::endl;
  error_= STSgetData(handle_, &header_,  (char*)buffer_, &frame_size_, 0, 0);
  // std::cerr<<"frame_size_:"<<frame_size_<<std::endl;
  if(!IsNoError("Failed to get data. "))  return false;
  return true;
}
//-------------------------------------------------------------------------------------------

//! Save RegMap to flash
bool TSentisM100::SaveToFlash()
{
  if(!PrintRegister(CmdExec,"CmdExec"))  return false;
  if(!PrintRegister(CmdExecResult,"CmdExecResult"))  return false;

  std::cerr<<"###Save RegMap to flash..."<<std::endl;
  std::cerr<<"Setting CmdExecPassword."<<std::endl;
  if(!WriteRegister(CmdExecPassword, 0x4877))  return false;
  std::cerr<<"Save RegMap to flash."<<std::endl;
  if(!WriteRegister(CmdExec, 0xDD9E))  return false;
  if(!PrintRegister(CmdExecResult,"CmdExecResult"))  return false;

  if(!PrintRegister(CmdExec,"CmdExec"))  return false;
  if(!PrintRegister(CmdExecResult,"CmdExecResult"))  return false;

  return true;
}
//-------------------------------------------------------------------------------------------

/*! Print a part of registers.
    \param [in] preset  0: major registers, 1: Ethernet related. */
bool TSentisM100::PrintRegisters(int preset=0)
{
  if(handle_==NULL)  return false;

  #define PRINT_REGISTER(x_address) \
    if(!PrintRegister(x_address, #x_address))  {return false;} \

  if(preset==0)
  {
    PRINT_REGISTER(Mode0)
    PRINT_REGISTER(Status)
    PRINT_REGISTER(ImageDataFormat)
    PRINT_REGISTER(IntegrationTime)  // min 50 max 25000
    PRINT_REGISTER(DeviceType)
    PRINT_REGISTER(DeviceInfo)
    PRINT_REGISTER(FirmwareInfo)
    PRINT_REGISTER(ModulationFrequency)
    PRINT_REGISTER(FrameRate)
    PRINT_REGISTER(HardwareConfiguration)
    PRINT_REGISTER(SerialNumberLowWord)
    PRINT_REGISTER(SerialNumberHighWord)
    PRINT_REGISTER(FrameCounter)
    PRINT_REGISTER(CalibrationCommand)
    PRINT_REGISTER(ConfidenceThresLow)
    PRINT_REGISTER(ConfidenceThresHig)
    PRINT_REGISTER(Mode1)
    PRINT_REGISTER(CalculationTime)
    PRINT_REGISTER(LedboardTemp)
    PRINT_REGISTER(MainboardTemp)
    PRINT_REGISTER(LinearizationAmplitude)
    PRINT_REGISTER(LinearizationPhasseShift)
    PRINT_REGISTER(FrameTime)
    PRINT_REGISTER(CalibrationExtended)
    PRINT_REGISTER(MaxLedTemp)
    PRINT_REGISTER(HorizontalFov)
    PRINT_REGISTER(VerticalFov)
    PRINT_REGISTER(TriggerDelay)
    PRINT_REGISTER(BootloaderStatus)
    PRINT_REGISTER(TemperatureCompensationGradient)
    PRINT_REGISTER(ApplicationVersion)
    PRINT_REGISTER(DistCalibGradient)
    PRINT_REGISTER(TempCompGradient2)
    PRINT_REGISTER(CmdExec)
    PRINT_REGISTER(CmdExecResult)
    PRINT_REGISTER(FactoryMacAddr2)
    PRINT_REGISTER(FactoryMacAddr1)
    PRINT_REGISTER(FactoryMacAddr0)
    PRINT_REGISTER(FactoryYear)
    PRINT_REGISTER(FactoryMonthDay)
    PRINT_REGISTER(FactoryHourMinute)
    PRINT_REGISTER(FactoryTimezone)
    PRINT_REGISTER(TempCompGradient3)
    PRINT_REGISTER(BuildYearMonth)
    PRINT_REGISTER(BuildDayHour)
    PRINT_REGISTER(BuildMinuteSecond)
    PRINT_REGISTER(UpTimeLow)
    PRINT_REGISTER(UpTimeHigh)
    PRINT_REGISTER(AkfPlausibilityCheckAmpLimit)
    PRINT_REGISTER(CommKeepAliveTimeout)
    PRINT_REGISTER(CommKeepAliveReset)
    PRINT_REGISTER(AecAvgWeight0)
    PRINT_REGISTER(AecAvgWeight1)
    PRINT_REGISTER(AecAvgWeight2)
    PRINT_REGISTER(AecAvgWeight3)
    PRINT_REGISTER(AecAvgWeight4)
    PRINT_REGISTER(AecAvgWeight5)
    PRINT_REGISTER(AecAvgWeight6)
    PRINT_REGISTER(AecAmpTarget)
    PRINT_REGISTER(AecTintStepMax)
    PRINT_REGISTER(AecTintMax)
    PRINT_REGISTER(AecKp)
    PRINT_REGISTER(AecKi)
    PRINT_REGISTER(AecKd)
  }
  else if(preset==1)
  {
    PRINT_REGISTER(Eth0Config)
    if(!PrintMacAddrRegister(Eth0Mac0, Eth0Mac1, Eth0Mac2, "Eth0Mac"))  return false;
    if(!PrintIPAddrRegister(Eth0Ip0, Eth0Ip1, "Eth0Ip"))  return false;
    if(!PrintIPAddrRegister(Eth0Snm0, Eth0Snm1, "Eth0Snm"))  return false;
    if(!PrintIPAddrRegister(Eth0Gateway0, Eth0Gateway1, "Eth0Gateway"))  return false;
    PRINT_REGISTER(Eth0TcpStreamPort)
    PRINT_REGISTER(Eth0TcpConfigPort)
    if(!PrintIPAddrRegister(Eth0UdpStreamIp0, Eth0UdpStreamIp1, "Eth0UdpStreamIp"))  return false;
    PRINT_REGISTER(Eth0UdpStreamPort)
  }

  #undef PRINT_REGISTER

  return true;
}
//-------------------------------------------------------------------------------------------


}
//-------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------
#include "nanotime.h"
#include <unistd.h>
//-------------------------------------------------------------------------------------------

using namespace std;
// using namespace boost;
using namespace loco_rabbits;
//-------------------------------------------------------------------------------------------
// #define print(var) PrintContainer((var), #var"= ")
// #define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  TSentisM100 tof_sensor;
  // tof_sensor.Init(/*init_fps=*/1, /*data_format=*/XYZ_COORDS_DATA, /*tcp_ip*/"192.168.111.21");
  tof_sensor.Init(/*init_fps=*/1, /*data_format=*/XYZ_COORDS_DATA, /*tcp_ip*/"192.168.0.14");
  std::cerr<<"----------"<<std::endl;
  // tof_sensor.PrintRegisters(0);
  std::cerr<<"----------"<<std::endl;
  tof_sensor.PrintRegisters(1);
  std::cerr<<"----------"<<std::endl;
  // tof_sensor.SetDHCP(false);
  std::cerr<<"----------"<<std::endl;
  // tof_sensor.SetDHCP(true);
  std::cerr<<"----------"<<std::endl;
  // tof_sensor.SaveToFlash();  return 0;
  std::cerr<<"----------"<<std::endl;
  int N= 400;
  tof_sensor.SetFrameRate(2);
  usleep(2000*1000);
  TTime TStart= GetCurrentTime();
  for(int i(0);i<N;++i)
  {
    std::cerr<<"Getting data "<<i<<"...";
    tof_sensor.GetData();
    std::cerr<<" obtained "<<tof_sensor.FrameSize()<<std::endl;
    usleep(25*1000);
  }
  std::cerr<<"----------"<<std::endl;
  double duration= (GetCurrentTime()-TStart).ToSec();
  std::cerr<<"Duration: "<<duration<<std::endl;
  std::cerr<<"FPS: "<<double(N)/duration<<std::endl;
  tof_sensor.Sleep();
  std::cerr<<"----------"<<std::endl;
  return 0;
}
//-------------------------------------------------------------------------------------------
