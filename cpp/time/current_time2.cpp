#include <iostream>
#include <iomanip>
#include <unistd.h>  // usleep
#include <sys/time.h>  // getrusage, gettimeofday

struct TTime
{
  long Sec;  // Seconds
  long USec;  // Micro-seconds
  TTime() : Sec(0), USec(0)  {}
  TTime(const long &s, const long &us) : Sec(s), USec(us)  {}
  double ToSec()  {return static_cast<double>(Sec) + static_cast<double>(USec)*1.0e-6l;}
  void Normalize()
    {
      long usec2= USec % 1000000L;
      long sec2= Sec + USec / 1000000L;
      if (usec2 < 0)
      {
        usec2+= 1000000L;
        --sec2;
      }
      Sec= sec2;
      USec= usec2;
    }
};
inline TTime operator+(const TTime &lhs, const TTime &rhs)
{
  TTime res(lhs);
  res.Sec+= rhs.Sec;
  res.USec+= rhs.USec;
  res.Normalize();
  return res;
}
inline TTime operator-(const TTime &lhs, const TTime &rhs)
{
  TTime res(lhs);
  res.Sec-= rhs.Sec;
  res.USec-= rhs.USec;
  res.Normalize();
  return res;
}
inline std::ostream& operator<<(std::ostream &lhs, const TTime &rhs)
{
  lhs<<"("<<rhs.Sec<<", "<<rhs.USec<<")";
  return lhs;
}

inline TTime GetCurrentTime(void)
{
  struct timeval time;
  gettimeofday (&time, NULL);
  TTime res;
  res.Sec= time.tv_sec;
  res.USec= time.tv_usec;
  return res;
}
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
    cout<<std::setprecision(18)<<GetCurrentTime().ToSec()<<"  "<<Test().ToSec()<<endl;
    // cout<<std::setprecision(18)<<GetCurrentTime().ToSec()<<"  "<<Test()<<endl;
    usleep(1000*100);
  }
  #endif
  return 0;
}



