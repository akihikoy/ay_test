//-------------------------------------------------------------------------------------------
/*! \file    ms-motion-recorder.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    Oct.21, 2013

    x++ -slora ms-motion-recorder.cpp
*/
//-------------------------------------------------------------------------------------------
// #include <lora/util.h>
// #include <lora/common.h>
// #include <lora/math.h>
// #include <lora/file.h>
// #include <lora/rand.h>
// #include <lora/small_classes.h>
// #include <lora/stl_ext.h>
#include <lora/serial.h>
// #include <lora/string.h>
// #include <lora/sys.h>
// #include <lora/octave.h>
// #include <lora/octave_str.h>
// #include <lora/ctrl_tools.h>
// #include <lora/ode.h>
// #include <lora/ode_ds.h>
// #include <lora/vector_wrapper.h>
#include <vector>
// #include <list>
// #include <boost/lexical_cast.hpp>
//-------------------------------------------------------------------------------------------
namespace loco_rabbits
{

namespace ms_motion_rec
{

// header part

//-------------------------------------------------------------------------------------------
// constants
//-------------------------------------------------------------------------------------------
static const unsigned char ACC2G    (0x01);
static const unsigned char ACC6G    (0x02);
static const unsigned char GYRO500  (0x21);
static const unsigned char GYRO2000 (0x22);
static const int    BUFFER_SIZE(2048);
//-------------------------------------------------------------------------------------------
enum TGravityOffsetKind {goInvalid=-1, goNone=0, goGravityX, goGravityY, goGravityZ};
//-------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------
// utilities
//-------------------------------------------------------------------------------------------
termios GetDefaultTermios (void);
//-------------------------------------------------------------------------------------------

//===========================================================================================
class TMotionRec
//===========================================================================================
{
public:
  struct TSample
    {
      // unsigned int Ch1: 10;
      // unsigned int Ch2: 10;
      // unsigned int Ch3: 10;
      // unsigned int Ch4: 10;
      // unsigned int Ch5: 10;
      // unsigned int Ch6: 10;
      // unsigned int Ch7: 10;
      // unsigned int Ch8: 10;
      short Ch1: 10;
      short Ch2: 10;
      short Ch3: 10;
      short Ch4: 10;
      short Ch5: 10;
      short Ch6: 10;
      short Ch7: 10;
      short Ch8: 10;
    };

  TMotionRec()  {}

  //! Connect to the sensor
  void Connect(const std::string &v_tty, const termios &v_ios=GetDefaultTermios());
  void Disconnect();

  bool IsConnected() const {return serial_.IsOpen();}

  //! Read to buffer_
  int Read()  {return Read(buffer_);}
  //! Read to buffer
  int Read(unsigned char *buffer)  {return serial_.Read(buffer, BUFFER_SIZE-1);}

  //! Read n-bytes to buffer_ (delay: usec)
  int ReadN(int n, int delay=10, int max_trial=10)  {return ReadN(n,buffer_,delay,max_trial);}
  //! Read n-bytes to buffer (delay: usec)
  int ReadN(int n, unsigned char *buffer, int delay=10, int max_trial=10);

  //! Send a command (string)
  void Send(const char *c)  {str_writes_<<c; str_writes_>>serial_;}

  //! Clear the serial buffer
  void Clear()  {Read();}

  //! Clear the serial buffer, send the reset command
  void Reset()  {Read(); Send("r"); usleep(100*1000);}

  const unsigned char* Buffer() const {return buffer_;}
  const unsigned char* SensorType() const {return sensor_type_;}
  const double* CalibrationValue() const {return calibration_value_;}
  const double* Sample() const {return curr_sample_;}

  std::vector<double>::const_iterator Sample(int i) const {return curr_samples_.begin()+i*8;}
  const std::vector<double>& Samples() const {return curr_samples_;}

  //! Return ID in string
  std::string GetID(int *num=NULL);

  //! Return sensor type in 8 byte binary
  const unsigned char* GetSensorType(int *num=NULL);

  //! Return the gravity offset kind used for the calibration (-1 for error)
  TGravityOffsetKind GetGravityOffsetKind(int *num=NULL);

  //! Return calibration value in double[8]
  const double* GetCalibrationValue(int *num=NULL);

  //! Set sensor type (8 byte binary)
  bool SetSensorType(const unsigned char type[8]);

  /*! Set sensor type (internal accelerometer type and internal gyro type);
      use ms_motion_rec::ACC2G or ACC6G for acc_type,
      ms_motion_rec::GYRO500 or GYRO2000 ro gyro_type  */
  bool SetSensorType(unsigned char acc_type, unsigned char gyro_type);

  /*! Calibrate the sensor */
  bool Calibrate(TGravityOffsetKind kind= goNone);

  //! Set sampling cycle (1-100 msec)
  bool SetSamplingCycle(int cycle);

  //! Start sampling
  void StartSampling();

  //! Get a sample
  const double* GetSample();

  //! Get n-samples
  const std::vector<double>& GetSamples(int n);

  //! Stop sampling
  bool StopSampling(int max_trial=20);

  double SensorCoefficient(int i)
    {
      switch(sensor_type_[i])
      {
      case ACC2G : return 0.0479;
      case ACC6G : return 0.1438;
      case GYRO500  : return REAL_PI/180.0*1.4663;
      case GYRO2000 : return REAL_PI/180.0*5.8651;
      }
      return 1.0;
    }

  template <typename t_out_itr>
  void ConvertObservation(const unsigned char dat[], t_out_itr res);

private:
  serial::TSerialCom  serial_;
  serial::TStringWriteStream   str_writes_;
  unsigned char buffer_[BUFFER_SIZE];

  unsigned char sensor_type_[8];
  double  calibration_value_[8];
  double  curr_sample_[8];
  std::vector<double>  curr_samples_;
};
//-------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------
}  // ms_motion_rec
}  // loco_rabbits
//-------------------------------------------------------------------------------------------


// source part

#include <sstream>
#include <iomanip>
#include <cstring>
//-------------------------------------------------------------------------------------------
namespace loco_rabbits
{
namespace ms_motion_rec
{
//-------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------
// utilities
//-------------------------------------------------------------------------------------------

termios GetDefaultTermios (void)
{
  termios ios;
  memset(&ios, 0, sizeof(ios));
  // ios.c_cflag |= B115200 | CREAD | CLOCAL | CS8;  /* control mode flags */
  ios.c_cflag |= B57600 | CREAD | CLOCAL | CS8;  /* control mode flags */
  ios.c_iflag |= IGNPAR;  /* input mode flags */
  ios.c_cc[VMIN] = 0;
  ios.c_cc[VTIME] = 1;  // *[1/10 sec] for timeout
  return ios;
}
//-------------------------------------------------------------------------------------------

//===========================================================================================
// class TMotionRec
//===========================================================================================

#define OS_BUFFER(x_n) "buffer("<<x_n<<")= "<<serial::PrintBuffer(buffer_,x_n)

void TMotionRec::Connect(const std::string &v_tty, const termios &v_ios)
{
  if(serial_.IsOpen())  serial_.Close();
  serial_.setting(v_tty,v_ios);
  serial_.Open();
}
//-------------------------------------------------------------------------------------------

void TMotionRec::Disconnect()
{
  if(serial_.IsOpen())  serial_.Close();
}
//-------------------------------------------------------------------------------------------

//! Read n-bytes to buffer (delay: usec)
int TMotionRec::ReadN(int n, unsigned char *buffer, int delay, int max_trial)
{
  int n_read(0), trial(0);
  for(trial=0; trial<max_trial; ++trial)
  {
    n_read+= serial_.Read(buffer+n_read, n-n_read);
    if(n_read>=n)  break;
    usleep(delay);
  }
  // LDBGVAR(trial);
  return n_read;
}
//-------------------------------------------------------------------------------------------

//! Return ID in string
std::string TMotionRec::GetID(int *num)
{
  Send("r");
  usleep(100*1000);
  Send("b");
  int n= ReadN(12);  if(num) *num= n;
  buffer_[n]='\0';
  Send("r");
  if(n!=12)  {LERROR("Failed: GetID"); LERROR(OS_BUFFER(n));}
  std::stringstream ss;
  ss<<buffer_;
  return ss.str();
}
//-------------------------------------------------------------------------------------------

//! Return sensor type in 8 byte binary
const unsigned char* TMotionRec::GetSensorType(int *num)
{
  Send("r");
  usleep(100*1000);
  Send("-");
  usleep(200*1000);
  Send("6o");
  int n= ReadN(8,sensor_type_);  if(num) *num= n;
  Send("r");
  if(n!=8)  {LERROR("Failed: GetSensorType"); LERROR(OS_BUFFER(n)); return NULL;}
  // cout<<serial::PrintBuffer(buffer_,n);
  return buffer_;
}
//-------------------------------------------------------------------------------------------

//! Return the gravity offset kind used for the calibration (-1 for error)
TGravityOffsetKind TMotionRec::GetGravityOffsetKind(int *num)
{
  Send("r");
  usleep(100*1000);
  Send("-");
  usleep(200*1000);
  Send("6p");
  int n= ReadN(1);  if(num) *num= n;
  Send("r");
  if(n==1)
  {
    switch(buffer_[0])
    {
    case 0x00: return goNone;
    case 0x01: return goGravityX;
    case 0x02: return goGravityY;
    case 0x04: return goGravityZ;
    }
  }
  LERROR("Failed: GetGravityOffsetKind");
  LERROR(OS_BUFFER(n));
  return goInvalid;
}
//-------------------------------------------------------------------------------------------

//! Return calibration value in double[8]
const double* TMotionRec::GetCalibrationValue(int *num)
{
  Send("r");
  usleep(100*1000);
  Send("-");
  usleep(200*1000);
  Send("6c");
  int n= ReadN(16);  if(num) *num= n;
  Send("r");
  if(n!=16)  {LERROR("Failed: GetCalibrationValue"); LERROR(OS_BUFFER(n)); return NULL;}
  for(int i(0); i<8; ++i)
  {
    calibration_value_[i]= 256.0*static_cast<double>(buffer_[i*2+0])+static_cast<double>(buffer_[i*2+1]);
    if(i<3)  calibration_value_[i]= 1023.0-calibration_value_[i];  // for internal accelerometer
  }
  // cout<<serial::PrintVector(calibration_value_,calibration_value_+8);
  return calibration_value_;
}
//-------------------------------------------------------------------------------------------

//! Set sensor type (8 byte binary)
bool TMotionRec::SetSensorType(const unsigned char type[8])
{
  Send("r");
  usleep(100*1000);
  Send("-");
  usleep(200*1000);
  Send("5o");
  usleep(200*1000);
  std::memcpy(sensor_type_,type,8);
  serial_.Write(sensor_type_,8);
  int n= ReadN(1);
  Send("r");
  if(n!=1 || buffer_[0]!='y')  {LERROR("Failed: SetSensorType"); LERROR(OS_BUFFER(n)); return false;}
  return true;
}
//-------------------------------------------------------------------------------------------

/*! Set sensor type (internal accelerometer type and internal gyro type);
    use ms_motion_rec::ACC2G or ACC6G for acc_type,
    ms_motion_rec::GYRO500 or GYRO2000 ro gyro_type  */
bool TMotionRec::SetSensorType(unsigned char acc_type, unsigned char gyro_type)
{
  unsigned char type[8];
  for(int i(0);i<3;++i)  type[i]= acc_type;
  type[3]= 0x00;
  for(int i(4);i<7;++i)  type[i]= gyro_type;
  type[7]= 0x00;
  return SetSensorType(type);
}
//-------------------------------------------------------------------------------------------

/*! Calibrate the sensor */
bool TMotionRec::Calibrate(TGravityOffsetKind kind)
{
  Send("r");
  usleep(100*1000);
  Send("-");
  usleep(200*1000);
  if(kind==goNone)
  {
    Send("4");
  }
  else
  {
    Send("9");
    usleep(200*1000);
    unsigned char d[1];
    switch(kind)
    {
    case goGravityX: d[0]= 0x01;  break;
    case goGravityY: d[0]= 0x02;  break;
    case goGravityZ: d[0]= 0x04;  break;
    default:
      LERROR("Error: invalid gravity offset kind"<<(int)kind);
      Send("r");
      return false;
    }
    serial_.Write(d,1);
  }
  int n= ReadN(1,500*1000);
  Send("r");
  if(n!=1 || buffer_[0]!='y')  {LERROR("Failed: Calibrate"); LERROR(OS_BUFFER(n)); return false;}
  return true;
}
//-------------------------------------------------------------------------------------------

//! Set sampling cycle (1-100 msec)
bool TMotionRec::SetSamplingCycle(int cycle)
{
  if(cycle<1)  cycle= 1;
  else if(cycle>100)  cycle= 100;
  Send("r");
  usleep(100*1000);
  Send("-");
  usleep(200*1000);
  Send("5d");
  usleep(200*1000);
  std::stringstream ss;
  ss<<std::setfill('0')<<std::setw(3)<<cycle;
  serial_.Write(ss.str().c_str(),3);
  int n= ReadN(1);
  Send("r");
  if(n!=1 || buffer_[0]!='y')  {LERROR("Failed: SetSamplingCycle"); LERROR(OS_BUFFER(n)); return false;}
  return true;
}
//-------------------------------------------------------------------------------------------

//! Start sampling
void TMotionRec::StartSampling()
{
  // preparation
  Send("r");
  usleep(100*1000);
  Send("-");
  usleep(200*1000);
  Send("1");
}
//-------------------------------------------------------------------------------------------

template <typename t_out_itr>
void TMotionRec::ConvertObservation(const unsigned char dat[], t_out_itr res)
{
  res[0]= ((dat[0]<<2))      + ((dat[1]&0xc0)>>6); // 8,2
  res[1]= ((dat[1]&0x3f)<<4) + ((dat[2]&0xf0)>>4); // 6,4
  res[2]= ((dat[2]&0x0f)<<6) + ((dat[3]&0xfc)>>2); // 4,6
  res[3]= ((dat[3]&0x03)<<8) + ((dat[4])); // 2,8

  res[4]= ((dat[5]<<2))      + ((dat[6]&0xc0)>>6); // 8,2
  res[5]= ((dat[6]&0x3f)<<4) + ((dat[7]&0xf0)>>4); // 6,4
  res[6]= ((dat[7]&0x0f)<<6) + ((dat[8]&0xfc)>>2); // 4,6
  res[7]= ((dat[8]&0x03)<<8) + ((dat[9])); // 2,8

  for(int i(0);i<3;++i)
    res[i]= 1023.0-res[i];

  for(int i(0);i<8;++i)
  {
    res[i]= SensorCoefficient(i)*(res[i]-calibration_value_[i]);
  }
}
//-------------------------------------------------------------------------------------------

//! Get a sample
const double* TMotionRec::GetSample()
{
  // TSample sample;
  // int n= ReadN(10, reinterpret_cast<unsigned char*>(&sample));
  int n= ReadN(10);
  if(n!=10)  {LERROR("Failed: GetSample; n="<<n);  return NULL;}

  ConvertObservation(buffer_, curr_sample_);

  #if 0
  curr_sample_[0]= ((buffer_[0]<<2))      + ((buffer_[1]&0xc0)>>6); // 8,2
  curr_sample_[1]= ((buffer_[1]&0x3f)<<4) + ((buffer_[2]&0xf0)>>4); // 6,4
  curr_sample_[2]= ((buffer_[2]&0x0f)<<6) + ((buffer_[3]&0xfc)>>2); // 4,6
  curr_sample_[3]= ((buffer_[3]&0x03)<<8) + ((buffer_[4])); // 2,8

  curr_sample_[4]= ((buffer_[5]<<2))      + ((buffer_[6]&0xc0)>>6); // 8,2
  curr_sample_[5]= ((buffer_[6]&0x3f)<<4) + ((buffer_[7]&0xf0)>>4); // 6,4
  curr_sample_[6]= ((buffer_[7]&0x0f)<<6) + ((buffer_[8]&0xfc)>>2); // 4,6
  curr_sample_[7]= ((buffer_[8]&0x03)<<8) + ((buffer_[9])); // 2,8

  for(int i(0);i<3;++i)
    curr_sample_[i]= 1023.0-curr_sample_[i];

  for(int i(0);i<3;++i)
  {
    switch(sensor_type_[i])
    {
    case ACC2G : curr_sample_[i]= 0.0479*(curr_sample_[i]-calibration_value_[i]); break;
    case ACC6G : curr_sample_[i]= 0.1438*(curr_sample_[i]-calibration_value_[i]); break;
    }
  }
  for(int i(4);i<7;++i)
  {
    switch(sensor_type_[i])
    {
    case GYRO500  : curr_sample_[i]= REAL_PI/180.0*1.4663*(curr_sample_[i]-calibration_value_[i]); break;
    case GYRO2000 : curr_sample_[i]= REAL_PI/180.0*5.8651*(curr_sample_[i]-calibration_value_[i]); break;
    }
  }
  #endif

  return curr_sample_;
}
//-------------------------------------------------------------------------------------------

//! Get n-samples
const std::vector<double>& TMotionRec::GetSamples(int n)
{
  if(n > (BUFFER_SIZE-1)/10)  n= (BUFFER_SIZE-1)/10;
  int n_read= ReadN(10*n);
  if(n_read!=10*n)
  {
    LERROR("Failed: GetSamples; n_read="<<n_read);
    curr_samples_.clear();
    return curr_samples_;
  }

  curr_samples_.resize(n*8);
  std::vector<double>::iterator s_itr(curr_samples_.begin());
  for(int i(0); i<n; ++i)
    ConvertObservation(buffer_+i*10, s_itr+i*8);
  return curr_samples_;
}
//-------------------------------------------------------------------------------------------

//! Stop sampling
bool TMotionRec::StopSampling(int max_trial)
{
  Read();  // clear the serial buffer
  int n(0);
  for(int trial(0); trial<max_trial; ++trial)
  {
    LDEBUG("sending z..");
    Send("z");
    n= Read();
    LDEBUG("n= "<<n);
    if(n>0 && buffer_[n-1]=='y')  break;
    usleep(500*1000);
  }
  Send("r");
  if(n==0 || buffer_[n-1]!='y')  {LERROR("Failed: StopSampling"); LERROR(OS_BUFFER(n)); return false;}
  return true;
}
//-------------------------------------------------------------------------------------------

#undef OS_BUFFER


}  // ms_motion_rec
}  // loco_rabbits
//-------------------------------------------------------------------------------------------


#ifndef NOT_MAIN
//-------------------------------------------------------------------------------------------
#include <lora/sys.h>
//-------------------------------------------------------------------------------------------
using namespace std;
// using namespace boost;
using namespace loco_rabbits;
//-------------------------------------------------------------------------------------------
// #define print(var) PrintContainer((var), #var"= ")
#define print(var) std::cerr<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  ms_motion_rec::TMotionRec motion_sensor;

  motion_sensor.Connect("/dev/rfcomm0");
  if(motion_sensor.IsConnected())
  {
    LMESSAGE("Connected");
  }
  else
  {
    LERROR("Failed to connect");
    return -1;
  }
  motion_sensor.Reset();

  print(motion_sensor.GetID());
  #if 1
  if(motion_sensor.GetSensorType())
    print(serial::PrintBuffer(motion_sensor.SensorType(),8));

  print((int)motion_sensor.GetGravityOffsetKind());
  if(motion_sensor.GetCalibrationValue())
    print(serial::PrintVector(motion_sensor.CalibrationValue(),motion_sensor.CalibrationValue()+8));

  /*! Set sensor type (internal accelerometer type and internal gyro type);
      use ms_motion_rec::ACC2G or ACC6G for acc_type,
      ms_motion_rec::GYRO500 or GYRO2000 ro gyro_type  */
  motion_sensor.SetSensorType(ms_motion_rec::ACC2G, ms_motion_rec::GYRO500);

  if(motion_sensor.GetSensorType())
    print(serial::PrintBuffer(motion_sensor.SensorType(),8));

  LMESSAGE("Calibrating..");
  motion_sensor.Calibrate(ms_motion_rec::goGravityZ);

  print((int)motion_sensor.GetGravityOffsetKind());
  if(motion_sensor.GetCalibrationValue())
    print(serial::PrintVector(motion_sensor.CalibrationValue(),motion_sensor.CalibrationValue()+8));
  #endif

  #if 0
  //! Set sampling cycle (1-100 msec)
  motion_sensor.SetSamplingCycle(100);
  motion_sensor.StartSampling();

  double t_start(GetCurrentTime());
  int n_total(0);
  // for(int i(0);i<100;++i)
  while(true)
  {
    const double *z= motion_sensor.GetSample();
    cout<<z[0]<<" "<<z[1]<<" "<<z[2]<<"  "<<z[4]<<" "<<z[5]<<" "<<z[6]<<endl;
    n_total+= 1;
    cerr<<GetCurrentTime()-t_start<<" "<<n_total<<endl;
    cerr<<"FPS: "<<(double)n_total/(GetCurrentTime()-t_start)<<endl;
    int c=KBHit();
    if(c=='x')  break;
  }
  cerr<<GetCurrentTime()-t_start<<" "<<n_total<<endl;
  cerr<<"FPS: "<<(double)n_total/(GetCurrentTime()-t_start)<<endl;
  #endif

  #if 1
  //! Set sampling cycle (1-100 msec)
  motion_sensor.SetSamplingCycle(5);
  motion_sensor.StartSampling();

  double t_start(GetCurrentTime());
  int n_total(0);
  double average[6]= {0.0,0.0,0.0, 0.0,0.0,0.0};
  // for(int i(0);i<100;++i)
  while(true)
  {
    const int NB(20);
    const std::vector<double> &z_set= motion_sensor.GetSamples(NB);
    if(static_cast<int>(z_set.size()/8)==NB)
    {
      for(std::vector<double>::const_iterator z(z_set.begin()),z_last(z_set.end()); z!=z_last; z+=8)
      {
        cout<<z[0]<<" "<<z[1]<<" "<<z[2]<<"  "<<z[4]<<" "<<z[5]<<" "<<z[6]<<endl;
        for(int r(0);r<3;++r)  average[r]+= z[r];
        for(int r(3);r<6;++r)  average[r]+= z[r+1];
      }
      n_total+= z_set.size()/8;
    }
    cerr<<GetCurrentTime()-t_start<<" "<<n_total<<endl;
    cerr<<"FPS: "<<(double)n_total/(GetCurrentTime()-t_start)<<endl;
    int c=KBHit();
    if(c=='x')  break;
  }
  cerr<<GetCurrentTime()-t_start<<" "<<n_total<<endl;
  cerr<<"FPS: "<<(double)n_total/(GetCurrentTime()-t_start)<<endl;
  cerr<<"Average: ";
  for(int r(0);r<6;++r)  cerr<<", "<<average[r]/(double)n_total;
  cerr<<endl;
  #endif

  motion_sensor.StopSampling();
  motion_sensor.Disconnect();
  return 0;
}
//-------------------------------------------------------------------------------------------

#endif // NOT_MAIN
