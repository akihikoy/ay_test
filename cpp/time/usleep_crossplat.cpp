//-------------------------------------------------------------------------------------------
/*! \file    usleep_crossplat.cpp
    \brief   Cross-platform usleep.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Aug.29, 2022

g++ usleep_crossplat.cpp
time ./a.out
*/
//-------------------------------------------------------------------------------------------
#include <iostream>
#include <chrono>
#include <thread>
//-------------------------------------------------------------------------------------------
#define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  int d_usec(300*1000);
  for(int i(0);i<10;++i)
  {
    std::cout<<i<<" "<<std::endl;
    std::this_thread::sleep_for(std::chrono::microseconds(d_usec));
  }
  return 0;
}
//-------------------------------------------------------------------------------------------
