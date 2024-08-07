//-------------------------------------------------------------------------------------------
/*! \file    cast_int.cpp
    \brief   Cast int to double.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Feb.26, 2018
*/
//-------------------------------------------------------------------------------------------
#include <iostream>
//-------------------------------------------------------------------------------------------
using namespace std;
//-------------------------------------------------------------------------------------------
// #define print(var) PrintContainer((var), #var"= ")
#define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  int a(10);
  print(a);
  print(a*0.2);
  print(int(a*0.2));
  print(int((a*0.02)*5));
  print(int((a*0.02)*5.0));
  return 0;
}
//-------------------------------------------------------------------------------------------
