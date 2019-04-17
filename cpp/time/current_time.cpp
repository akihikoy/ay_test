#include <iostream>
#include <iomanip>
#include <unistd.h>  // usleep
#include <sys/time.h>  // getrusage, gettimeofday
typedef double t_real;
// typedef long double t_real;
inline t_real GetCurrentTime(void)
{
  struct timeval time;
  gettimeofday (&time, NULL);
  return static_cast<t_real>(time.tv_sec) + static_cast<t_real>(time.tv_usec)*1.0e-6l;
}
t_real Test(void)
{
  t_real t_start= GetCurrentTime();
  double x(0);
  for(int i(0);i<10;++i)
    x+=i;  // some computation
  return GetCurrentTime()-t_start;
}
using namespace std;
int main()
{
  #define TEST1
  // #define TEST2
  #ifdef TEST1
  int i(0);
  t_real t_start= GetCurrentTime();
  t_real t(0.0l);
  while(true)
  {
    t= GetCurrentTime();
    while(GetCurrentTime()-t<1.0l)
    {
      ++i;
      // cout<<std::setprecision(18)<<GetCurrentTime()<<"  "<<GetCurrentTime()-t<<"  "<<i<<endl;
    }
    // cout<<"------------"<<endl;
    cout<<std::setprecision(18)<<GetCurrentTime()<<"  "<<GetCurrentTime()-t_start<<"  "<<i<<endl;
  }
  #endif
  #ifdef TEST2
  while(true)
  {
    cout<<std::setprecision(18)<<GetCurrentTime()<<"  "<<Test()<<endl;
    usleep(1000*100);
  }
  #endif
  return 0;
}



