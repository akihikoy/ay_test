//-------------------------------------------------------------------------------------------
/*! \file    darwin_mini01.cpp
    \brief   Darwin mini test code
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \date    Jan 10, 2015
*/
//-------------------------------------------------------------------------------------------
#include <lora/bioloid.h>
#include <lora/stl_ext.h>
#include <lora/sys.h>
#include <lora/rand.h>
#include <fstream>
#include <csignal>
//-------------------------------------------------------------------------------------------
namespace loco_rabbits
{

template <typename t_angle_fwditr>
inline void ConstrainAngles (t_angle_fwditr begin, t_angle_fwditr end)
{
  // const double angle_max[18]={ 0.0, 1.6, 1.6, 1.6, 1.6, 1.5,   1.57, 1.57, 1.6, 1.6, 1.6, 1.5,   1.6, 0.0, 1.6, 1.6, 1.6, 1.5};
  // const double angle_min[18]={-1.6,-0.0,-1.6,-1.6,-1.5,-1.6,  -1.57,-1.57,-1.6,-1.6,-1.5,-1.6,  -0.0,-1.6,-1.6,-1.6,-1.5,-1.6};
  const double angle_max[18]={  0,90, 90, 90, 90, 80,   90, 90, 90, 90, 90, 80,   90,  0, 90, 90, 90, 80};
  const double angle_min[18]={-90,-0,-90,-90,-80,-90,  -90,-90,-90,-90,-80,-90,   -0,-90,-90,-90,-80,-90};
  const double *amax(angle_max), *amin(angle_min);
  for (; begin!=end; ++begin,++amax,++amin)
  {
    if(*begin>*amax)  *begin= *amax;
    else if (*begin<*amin)  *begin= *amin;
  }
}

}
//-------------------------------------------------------------------------------------------
using namespace std;
using namespace loco_rabbits;
using namespace bioloid;
//-------------------------------------------------------------------------------------------

const char *BIOL_PORT("/dev/ttyUSB0");

static bool Executing(true);
static TBioloidController *PtrBioloidCtrl(NULL);
static bool ReadAngleSignal(false);
static bool ReadVAngleSignal(false);
static bool RandomAngles(false);

void sig_handler(int signo)
{
  using namespace std;
  if(!Executing)
    {std::cerr<<"quit immediately!"<<std::endl;  exit(1);}

  if(signo==SIGINT)
  {
    std::cerr<<ioscc::blue<<"interrupted."<<std::endl;
    std::cerr<<
      "  'A':  read all angles"<<std::endl<<
      "  'V':  read virtual angles"<<std::endl<<
      "  'I':  initialize communication"<<std::endl<<
      "  'G':  switch random/sin angles"<<std::endl<<
      "  Space:  continue"<<std::endl<<
      "  'Q':  quit"<<std::endl;
    while(true)
    {
      std::cerr<<"  > "<<std::flush;
      int res= WaitKBHit();
      if(res=='q' || res=='Q')
      {
        Executing=false;
        break;
      }
      else if(res==' ')
      {
        Executing=true;
        break;
      }
      else if(res=='a' || res=='A')
      {
        ReadAngleSignal=true;
        break;
      }
      else if(res=='v' || res=='V')
      {
        ReadVAngleSignal=true;
        break;
      }
      else if(res=='i' || res=='I')
      {
        // PtrBioloidCtrl->Connect(BIOL_PORT);
        // PtrBioloidCtrl->TossMode();
        PtrBioloidCtrl->ConnectBS(BIOL_PORT);
        break;
      }
      else if(res=='g' || res=='G')
      {
        RandomAngles= !RandomAngles;
        break;
      }
      else {std::cerr<<"unknown action."<<std::endl;}
    }
  }
  else if (signo==SIGQUIT)
  {
    std::cerr<<"quit..."<<std::endl;
    Executing=false;
  }
  else  std::cerr<<"signal code "<<signo<<" is ignored."<<std::endl;
}
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  signal(SIGINT,sig_handler);
  signal(SIGQUIT,sig_handler);

  // termios ios= GetDefaultTermios();
  // ios.c_cflag = B57600 | CREAD | CLOCAL | CS8;  /* control mode flags */

  TBioloidController  bioloid;
  // bioloid.Connect(BIOL_PORT);
  // bioloid.TossMode();
  // // bioloid.TossTest();
  bioloid.ConnectBS(BIOL_PORT);

  PtrBioloidCtrl= &bioloid;


  #if 0
  bioloid.SetLightDetectCompare (100,128);
  #endif

  // int ids[]= {1,2};
  int ids[]= {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18};
  double angles[SIZE_OF_ARRAY(ids)], angles_observed[SIZE_OF_ARRAY(ids)];

  ofstream ofs_goal("res/goal.dat");
  ofstream ofs_angle("res/angle.dat");
  ofstream ofs_vangle("res/vangle.dat");
  int skip(0);
  double goal, distance_c;
  double time_offset(GetCurrentTime()), time(GetCurrentTime()), t;
  int    disc_time(0);

  #if 0
  while (time-time_offset<2.0)
  {
    bioloid.GetAllAngles (ids,ids+SIZE_OF_ARRAY(ids), angles_observed);
    bioloid.GoTo (ids,ids+SIZE_OF_ARRAY(ids), angles);
    usleep(10000);
    time= GetCurrentTime();
  }
  #endif
  LMESSAGE("----");

  while (Executing)
  {
    t= time-time_offset;
    if (!RandomAngles)
    {
      goal= 30.0*sin(2.0*M_PI*0.2*t);
      LMESSAGE(t<<"[s]: ("<<goal<<") ");
      std::fill (angles,angles+SIZE_OF_ARRAY(angles),goal);
    }
    else
    {
      LMESSAGE(t<<"[s]");
      if (disc_time%10==0)
      {
        for (int j(0); j<SIZE_OF_ARRAY(angles); ++j)
          angles[j]= Rand(-60.0,60.0);
      }
    }

    #if 1
    // bioloid.GetAngle(1);
    if(skip==0)
    {
      #if 0
      bioloid.GetAllAngles (ids,ids+SIZE_OF_ARRAY(ids), angles_observed);
      ofs_angle<<t<<" "<<ContainerToStr(angles_observed,angles_observed+SIZE_OF_ARRAY(angles_observed))<<endl;
      // bioloid.GetAllAngles (ids,ids+5, angles_observed);
      // ofs_angle<<t<<" "<<ContainerToStr(angles_observed,angles_observed+5)<<endl;
      #endif
      #if 0
      bioloid.GetVirtualAngles (ids,ids+SIZE_OF_ARRAY(ids), angles_observed);
      ofs_vangle<<t<<" "<<ContainerToStr(angles_observed,angles_observed+SIZE_OF_ARRAY(angles_observed))<<endl;
      #endif
      skip=0;
    } else{--skip;}

    if(ReadAngleSignal)
    {
      PtrBioloidCtrl->GetAllAngles (ids,ids+SIZE_OF_ARRAY(ids), angles_observed);
      cout<<" "<<ContainerToStr(angles_observed,angles_observed+SIZE_OF_ARRAY(angles_observed))<<endl;
      ReadAngleSignal=false;
    }

    if(ReadVAngleSignal)
    {
      PtrBioloidCtrl->GetAllAngles (ids,ids+SIZE_OF_ARRAY(ids), angles_observed);
      cout<<" "<<ContainerToStr(angles_observed,angles_observed+SIZE_OF_ARRAY(angles_observed))<<endl;

      PtrBioloidCtrl->GetVirtualAngles (ids,ids+SIZE_OF_ARRAY(ids), angles_observed);
      cout<<" "<<ContainerToStr(angles_observed,angles_observed+SIZE_OF_ARRAY(angles_observed))<<endl;
      ReadVAngleSignal=false;
    }

    ConstrainAngles (angles,angles+SIZE_OF_ARRAY(angles));
    bioloid.GoTo (ids,ids+SIZE_OF_ARRAY(ids), angles);
    ofs_goal<<t<<" "<<ContainerToStr(angles,angles+SIZE_OF_ARRAY(angles))<<endl;
    #endif

    #if 0
    bioloid.GetDistance(100,0,distance_c);
    ofs_angle<<t<<" "<<distance_c<<endl;
    #endif

    // usleep(5000);
//     usleep(10000);
    // usleep(15000);
    // usleep(20000);
    // usleep(30000);
    // usleep(50000);
    usleep(100000);
    time= GetCurrentTime();
    ++disc_time;
  }

  ofs_goal.close();
  ofs_angle.close();
  ofs_vangle.close();

  bioloid.Disconnect();

  return 0;
}
//-------------------------------------------------------------------------------------------
