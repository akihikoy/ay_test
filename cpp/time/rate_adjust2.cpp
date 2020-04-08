//-------------------------------------------------------------------------------------------
/*! \file    rate_adjust2.cpp
    \brief   Rate control test to see if it has nano-sec accuracy.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Apr.08, 2020

$ g++ -O3 rate_adjust2.cpp
$ ./a.out
000000000 000000357
000000000 500104044
000000001 000095330
000000001 500094518
000000002 000095158
000000002 500096613
000000003 000094109
000000003 500160986
000000004 000124733
000000004 500091583
000000005 000127616
000000005 500074904
000000006 000123788
000000006 500094189
000000007 000143448
000000007 500076189
000000008 000129132
000000008 500093879
000000009 000101642
000000009 500072084
MAX ERROR: 160986 ns.
*/
//-------------------------------------------------------------------------------------------
#include <iostream>
#include <iomanip>
#define LIBRARY
#include "clock_gettime.cpp"
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  TTime t0= GetCurrentTime();
  int FPS(20);
  long interval= 1000000000L/FPS;
  for(int i(0);;++i)
  {
    TTime t1= GetCurrentTime();
    std::cerr<<std::setfill('0')<<std::setw(9)<<(t1-t0).Sec<<" "<<std::setfill('0')<<std::setw(9)<<(t1-t0).NSec<<std::endl;
    TTime dt= (t0+TTime(0,(i+1)*interval))-GetCurrentTime();
    usleep(dt.Sec*1000000+dt.NSec/1000);
  }
  return 0;
}
//-------------------------------------------------------------------------------------------
