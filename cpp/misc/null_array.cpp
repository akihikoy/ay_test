//-------------------------------------------------------------------------------------------
/*! \file    null_array.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Feb.11, 2015
*/
//-------------------------------------------------------------------------------------------
#include <iostream>
//-------------------------------------------------------------------------------------------
void Func1(double x[1])
{
  if(x) x[0]*= 2.0;
  else std::cerr<<"x is NULL"<<std::endl;
}
void Func2(double x[1]=NULL)
{
  if(x) x[0]*= 3.0;
  else std::cerr<<"x is NULL"<<std::endl;
}
//-------------------------------------------------------------------------------------------
using namespace std;
//-------------------------------------------------------------------------------------------
// #define print(var) PrintContainer((var), #var"= ")
#define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  double x[1]= {1.2};
  print(x[0]);
  Func1(x);
  print(x[0]);
  Func1(NULL);
  print(x[0]);
  Func1(x);
  print(x[0]);
  Func2(x);
  print(x[0]);
  Func2();
  print(x[0]);
  Func2(x);
  print(x[0]);
  return 0;
}
//-------------------------------------------------------------------------------------------
