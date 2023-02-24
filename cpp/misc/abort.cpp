//-------------------------------------------------------------------------------------------
/*! \file    abort.cpp
    \brief   Abort the execution to generate core-dump on purpose.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Feb.24, 2023

g++ -Wall -O2 abort.cpp -o abort.out
*/
//-------------------------------------------------------------------------------------------
#include <iostream>
int main(int argc, char**argv)
{
  std::cout<<"p1"<<std::endl;
  abort();
  std::cout<<"p2"<<std::endl;
  return 0;
}
//-------------------------------------------------------------------------------------------
