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
