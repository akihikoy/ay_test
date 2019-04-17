//-------------------------------------------------------------------------------------------
/*! \file    vector_of_obj.cpp
    \brief   For vector of a class when constructor and destructor are called?
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    May.05, 2017

g++ -g -Wall vector_of_obj.cpp
*/
//-------------------------------------------------------------------------------------------
#include <iostream>
#include <vector>
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
// #define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

class TTest
{
public:
  int I;
  TTest()
    {
      std::cerr<<"TTest constructor is called "<<I<<" @"<<reinterpret_cast<void*>(this)<<std::endl;
    }
  ~TTest()
    {
      std::cerr<<"TTest destructor is called "<<I<<" @"<<reinterpret_cast<void*>(this)<<std::endl;
      I= 0;
    }
};

int main(int argc, char**argv)
{
  std::cerr<<"1"<<std::endl;
  std::vector<TTest>  vec(2);
  vec[0].I=10; vec[1].I=20;
  std::cerr<<"2"<<std::endl;
  vec.clear();
  std::cerr<<"3"<<std::endl;
  vec.resize(2);
  vec[0].I=100; vec[1].I=200;
  std::cerr<<"4"<<std::endl;
  return 0;
}
//-------------------------------------------------------------------------------------------
