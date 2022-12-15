//-------------------------------------------------------------------------------------------
/*! \file    default-array-val2.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Dec.15, 2022
*/
//-------------------------------------------------------------------------------------------
#include <iostream>
//-------------------------------------------------------------------------------------------
using namespace std;
//-------------------------------------------------------------------------------------------
// #define print(var) PrintContainer((var), #var"= ")
#define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

// void Print(const int array[5]={10,20,30,40,50})  // NOTE: THIS IS IMPOSSIBLE(COMPILE ERROR)
void Print(const int array[5]=NULL)
{
  static const int default_array[]= {10,20,30,40,50};
  if(array==NULL)  array= default_array;
  for(int i(0); i<5; ++i)
  {
    print(array[i]);
  }
}

int main(int argc, char**argv)
{
  Print();
  int x[]= {50,40,30,20,10};
  Print(x);
  return 0;
}
//-------------------------------------------------------------------------------------------
