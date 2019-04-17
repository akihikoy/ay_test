//-------------------------------------------------------------------------------------------
/*! \file    default-array-val.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    Aug.04, 2014
*/
//-------------------------------------------------------------------------------------------
#include <iostream>
//-------------------------------------------------------------------------------------------
using namespace std;
//-------------------------------------------------------------------------------------------
// #define print(var) PrintContainer((var), #var"= ")
#define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

const int DefaultArray[]= {1,2,3,4,5};

// void Print(const int array[5]={1,2,3,4,5})  // NOTE: THIS IS IMPOSSIBLE(COMPILE ERROR)
void Print(const int array[5]=DefaultArray)
{
  for(int i(0); i<5; ++i)
  {
    print(array[i]);
  }
}

int main(int argc, char**argv)
{
  Print();
  int x[]= {5,4,3,2,1};
  Print(x);
  return 0;
}
//-------------------------------------------------------------------------------------------
