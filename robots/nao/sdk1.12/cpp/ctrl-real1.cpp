//-------------------------------------------------------------------------------------------
/*! \file    ctrl-real1.cpp
    \brief   test program to control the real nao
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    May.13, 2011
*/
//-------------------------------------------------------------------------------------------
#include <alproxy.h>
// #include <alproxies/almotionproxy.h>
//-------------------------------------------------------------------------------------------
namespace loco_rabbits
{
}
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

  try
  {
    const char IP[]="163.221.139.150";
    int port=9559;
            // we have to explicitly write IP and port because brokers are not connected

    // AL::ALPtr<AL::ALProxy> audioProxy(new AL::ALProxy("ALTextToSpeech",IP,port));
    // audioProxy->callVoid("say",string("test pattern"));


    AL::ALPtr<AL::ALProxy> motion(new AL::ALProxy("ALMotion",IP,port));
            // pointer automatically destroyed after try block
            // or ALProxy *motion = new ALProxy("ALMotion",IP,port);
            // don't forget to delete proxy manually

    {
      // Example showing how to interpolate to maximum stiffness in 1 second
      string names  = "Body";
      double stiffnessLists  = 1.0;  // NOTE: it seems not working in Choregraphe
      double timeLists  = 1.0;
      motion->callVoid("stiffnessInterpolation", names, stiffnessLists, timeLists);
    }

    {
      // Example showing how to set angles, using a fraction of max speed
      vector<string> tmp_names;
      tmp_names.push_back("HeadYaw");
      tmp_names.push_back("HeadPitch");
      AL::ALValue names(tmp_names);

      vector<float> tmp_angles;
      tmp_angles.push_back(-1.0);
      tmp_angles.push_back(-0.2);
      AL::ALValue angles(tmp_angles);

      double fractionMaxSpeed  = 0.2;
      motion->callVoid("setAngles", names, angles, fractionMaxSpeed);
    }

    {
      // Example showing a single target angle for one joint
      // Interpolate the head yaw to 1.0 radian in 1.0 second
      vector<string> tmp_names;
      tmp_names.push_back("HeadYaw");
      tmp_names.push_back("HeadPitch");
      AL::ALValue names(tmp_names);

      AL::ALValue angles;
      angles.arraySetSize(2);
      angles[0] = AL::ALValue::array(1.0f, 0.0f);
      angles[1] = AL::ALValue::array(-0.5f, 0.5f, 0.0f);

      AL::ALValue times;
      times.arraySetSize(2);
      times[0] = AL::ALValue::array(1.0f, 2.0f);
      times[1] = AL::ALValue::array(1.0f, 2.0f, 3.0f);

      bool isAbsolute  = true;
      motion->callVoid("angleInterpolation", names, angles, times, isAbsolute);
    }

  }
  catch(...)
  {
  }

  return 0;
}
//-------------------------------------------------------------------------------------------
