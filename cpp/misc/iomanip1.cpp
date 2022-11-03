//-------------------------------------------------------------------------------------------
/*! \file    iomanip1.cpp
    \brief   Test of iomanip
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Nov.03, 2022

g++ -O2 -Wall iomanip1.cpp
*/
//-------------------------------------------------------------------------------------------
#include <iostream>
#include <sstream>
#include <iomanip>
//-------------------------------------------------------------------------------------------
#define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  {
    std::stringstream ss;
    ss<<"test"<<std::setprecision(10)<<3.14;
    print(ss.str());
  }
  {
    std::stringstream ss;
    ss<<"test"<<std::setfill('0')<<std::setw(5)<<12;
    print(ss.str());
  }
  {
    std::stringstream ss;
    ss<<"test"<<std::setfill('0')<<std::setw(10)<<std::setprecision(5)<<3.14;
    print(ss.str());
  }
  return 0;
}
//-------------------------------------------------------------------------------------------
