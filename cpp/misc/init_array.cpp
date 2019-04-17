//-------------------------------------------------------------------------------------------
/*! \file    init_array.cpp
    \brief   Array initializations
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Apr.04, 2016
*/
//-------------------------------------------------------------------------------------------
#include <iostream>
#include <string>
//-------------------------------------------------------------------------------------------
namespace loco_rabbits
{
}
//-------------------------------------------------------------------------------------------
using namespace std;
// using namespace boost;
using namespace loco_rabbits;
//-------------------------------------------------------------------------------------------
// #define print(var) PrintContainer((var), #var"= ")
#define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  std::string sarray[2]= {"aaa", "bbb"};
  print(sarray[0]);
  print(sarray[1]);
  return 0;
}
//-------------------------------------------------------------------------------------------
