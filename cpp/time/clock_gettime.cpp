/*
src:
  https://github.com/ros/roscpp_core/blob/indigo-devel/rostime/src/time.cpp
*/
#include <iostream>
#include <iomanip>
#include <unistd.h>  // usleep
#include <ctime>  // clock_gettime
#include <sys/time.h>  // getrusage, gettimeofday
struct TTime
{
  long Sec;  // Seconds
  long NSec;  // Nano-seconds
  TTime() : Sec(0), NSec(0)  {}
  TTime(const long &s, const long &ns) : Sec(s), NSec(ns)  {}
  double ToSec()  {return static_cast<double>(Sec) + static_cast<double>(NSec)*1.0e-9l;}
  void Normalize()
    {
      long nsec2= NSec % 1000000000L;
      long sec2= Sec + NSec / 1000000000L;
      if (nsec2 < 0)
      {
        nsec2+= 1000000000L;
        --sec2;
      }
      Sec= sec2;
      NSec= nsec2;
    }
};
inline TTime operator+(const TTime &lhs, const TTime &rhs)
{
  TTime res(lhs);
  res.Sec+= rhs.Sec;
  res.NSec+= rhs.NSec;
  res.Normalize();
  return res;
}
inline TTime operator-(const TTime &lhs, const TTime &rhs)
{
  TTime res(lhs);
  res.Sec-= rhs.Sec;
  res.NSec-= rhs.NSec;
  res.Normalize();
  return res;
}
inline std::ostream& operator<<(std::ostream &lhs, const TTime &rhs)
{
  lhs<<"("<<rhs.Sec<<", "<<rhs.NSec<<")";
  return lhs;
}

#define HAS_CLOCK_GETTIME (_POSIX_C_SOURCE >= 199309L)
inline TTime GetCurrentTime(void)
{
#if HAS_CLOCK_GETTIME
  timespec start;
  clock_gettime(CLOCK_REALTIME, &start);
  TTime res;
  res.Sec= start.tv_sec;
  res.NSec= start.tv_nsec;
  return res;
#else
  struct timeval time;
  gettimeofday (&time, NULL);
  TTime res;
  res.Sec= time.tv_sec;
  res.NSec= time.tv_usec*1000L;
  return res;
#endif
}

#ifndef LIBRARY
TTime Test(void)
{
  TTime t_start= GetCurrentTime();
  double x(0);
  for(int i(0);i<10;++i)
    x+=i;  // some computation
  return GetCurrentTime()-t_start;
}
using namespace std;
int main()
{
  // #define TEST1
  #define TEST2
  #ifdef TEST1
  int i(0);
  TTime t_start= GetCurrentTime();
  TTime t(0,0);
  while(true)
  {
    t= GetCurrentTime();
    while((GetCurrentTime()-t).ToSec()<1.0l)
    {
      ++i;
      // cout<<std::setprecision(18)<<GetCurrentTime().ToSec()<<"  "<<(GetCurrentTime()-t).ToSec()<<"  "<<i<<endl;
    }
    // cout<<"------------"<<endl;
    cout<<std::setprecision(18)<<GetCurrentTime().ToSec()<<"  "<<(GetCurrentTime()-t_start).ToSec()<<"  "<<i<<endl;
  }
  #endif
  #ifdef TEST2
  while(true)
  {
    // cout<<std::setprecision(18)<<GetCurrentTime().ToSec()<<"  "<<Test().ToSec()<<endl;
    cout<<std::setprecision(18)<<GetCurrentTime().ToSec()<<"  "<<Test()<<endl;
    usleep(1000*100);
  }
  #endif
  return 0;
}
#endif//LIBRARY


