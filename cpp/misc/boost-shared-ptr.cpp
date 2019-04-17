//-------------------------------------------------------------------------------------------
/*! \file    boost-shared-ptr.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    Aug.04, 2014
*/
//-------------------------------------------------------------------------------------------
#include <iostream>
#include <unistd.h>
// #include <string>
// #include <vector>
// #include <list>
#include <boost/shared_ptr.hpp>
//-------------------------------------------------------------------------------------------
using namespace std;
using namespace boost;
//-------------------------------------------------------------------------------------------
// #define print(var) PrintContainer((var), #var"= ")
// #define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

class TTest
{
public:
  double Data[100];
  TTest(int c)
    {
      for(int i(0); i<100; ++i)  Data[i]= c;
    }
};

void Func1(boost::shared_ptr<TTest> &ptr, int c)
{
  ptr= boost::shared_ptr<TTest>(new TTest(c));
}

int main(int argc, char**argv)
{
  boost::shared_ptr<TTest> ptr;
  for(int i(0); i<100; ++i)
  {
    Func1(ptr, 1);
    // usleep(300*1000);
    for(int a(0); a<1000000; ++a)
      ptr->Data[a%100]= a;
  }
  return 0;
}
//-------------------------------------------------------------------------------------------
