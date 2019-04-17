#include <bioloid.h>
#include <fstream>
#include <cstdlib>
#include <cmath>

using namespace std;
using namespace loco_rabbits;
using namespace serial;
using namespace bioloid;
//-------------------------------------------------------------------------------------------
// #define print(var) print_container((var), #var"= ")
// #define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

template<typename T>
inline T URand()
{
  return static_cast<T>(std::rand())/static_cast<T>(RAND_MAX);
}
template<typename T>
inline T URand(const T &min,const T &max)
{
  return (max-min)*URand<T>()+min;
}

int main(int argc, char**argv)
{
  TBioloidController  bioloid;

  bioloid.Connect("/dev/ttyUSB0");
  bioloid.TossMode();
  // // bioloid.TossTest();

  // bioloid.ConnectBS("/dev/ttyUSB0",1);

  int ids[]= {11};
  double angles[SIZE_OF_ARRAY(ids)];
  // double angles_observed[SIZE_OF_ARRAY(ids)];

  LMESSAGE("----");

  srand((unsigned)time(NULL));

  ofstream ofs_param("biof/fing_param.dat", std::ios::out | std::ios::app);
  ofstream ofs_dist("biof/fing_dist.dat", std::ios::out | std::ios::app);

  while(true)
  {
    string buf;

    // swing back
    angles[0]= -90.0;
    bioloid.GoTo(ids,ids+SIZE_OF_ARRAY(ids), angles);
    cout<<" Put the ball on the finger, then type something and press return (q to exit) > ";
    cin>>buf;
    if(buf=="q" || buf=="Q")  break;

    // play swing
    double Param[2];
    Param[0]= URand(0.001, 3.14*2.0);
    Param[1]= URand(-2.0, 3.0);
    double time_offset(GetCurrentTime());
    while(angles[0]<Param[1]*10.0)
    {
      double t= GetCurrentTime() - time_offset;
      angles[0]= -90.0 + t*180.0*(1.0-cos(Param[0]*t))*0.5;
      // cerr<<t<<"  "<<angles[0]<<endl;
      bioloid.GoTo(ids,ids+SIZE_OF_ARRAY(ids), angles);
      usleep(20000);
    }

    // log result
    cout<<" Measure the ball position > ";
    double dist(0.0);
    cin>>dist;
    cout<<Param[0]<<" "<<Param[1]<<" "<<dist<<endl;
    ofs_param<<Param[0]<<" "<<Param[1]<<endl;
    ofs_dist<<dist<<endl;
  }

  bioloid.Disconnect();

  return 0;
}
//-------------------------------------------------------------------------------------------
