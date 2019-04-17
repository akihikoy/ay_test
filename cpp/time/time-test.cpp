#include <iostream>
#include <sys/time.h>  // getrusage, gettimeofday
#include <sys/resource.h> // get cpu time
inline double GetUserTime (void)
{
  struct rusage RU;
  getrusage(RUSAGE_SELF, &RU);
  return static_cast<double>(RU.ru_utime.tv_sec) + static_cast<double>(RU.ru_utime.tv_usec)*1.0e-6;
}
using namespace std;
int main()
{
  int i(0);
  double t_start= GetUserTime();
  double t(0.0);
  while(true)
  {
    t= GetUserTime();
    while(GetUserTime()-t<1.0)
    {
      ++i;
// cout<<GetUserTime()-t<<"  "<<i<<endl;
    }
cout<<"------------"<<endl;
    cout<<GetUserTime()-t_start<<"  "<<i<<endl;
  }
  return 0;
}



