//-------------------------------------------------------------------------------------------
/*! \file    median.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    Jul.01, 2014
*/
//-------------------------------------------------------------------------------------------
#include <iostream>
// #include <iomanip>
#include <algorithm>
#include <vector>
// #include <list>
// #include <boost/lexical_cast.hpp>
//-------------------------------------------------------------------------------------------
namespace loco_rabbits
{
}
//-------------------------------------------------------------------------------------------
// using namespace std;
// using namespace boost;
using namespace loco_rabbits;
//-------------------------------------------------------------------------------------------
// #define print(var) PrintContainer((var), #var"= ")
// #define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  // int src[] = {2,10,1};
  // int src[] = {2,10,5,1};
  int src[] = {2,10,5,1,55,7,48,103,22,6,3,99,45,99};


  std::vector<int> array(src,src+sizeof(src)/sizeof(src[0]));

  std::sort(array.begin(),array.end());
  std::cout<<"median: "<<array[array.size()/2]<<std::endl;

  return 0;
}
//-------------------------------------------------------------------------------------------
