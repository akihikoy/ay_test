#include <iostream>
#include <iomanip>
#include <unistd.h>  // usleep
#include <sys/time.h>  // getrusage, gettimeofday
typedef double t_real;
// typedef long double t_real;
inline long GetCurrentTime(void)
{
  struct timeval time;
  gettimeofday (&time, NULL);
  return time.tv_sec*1e6l + time.tv_usec;
}
inline timeval GetCurrentTime2(void)
{
  struct timeval time;
  gettimeofday (&time, NULL);
  return time;
}
using namespace std;
int main()
{
  long t_start= GetCurrentTime();
  timeval t_start2= GetCurrentTime2();
  cout<<"start at: "<<t_start<<", "<<t_start2.tv_sec<<"+"<<t_start2.tv_usec<<std::endl;
  for(int i(0); i<10; ++i)
  {
    long t= GetCurrentTime();
    timeval t2= GetCurrentTime2();
    cout<<"start at: "<<t<<", "<<t2.tv_sec<<"+"<<t2.tv_usec<<std::endl;
    cout<<"  equal?"<<(t==t_start)/*<<", "<<(t2==t_start2)*/<<std::endl;
  }
  return 0;
}



