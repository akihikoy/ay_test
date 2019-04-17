//-------------------------------------------------------------------------------------------
/*! \file    friend-template.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    Dec.07, 2011
*/
//-------------------------------------------------------------------------------------------
#include <iostream>

class TTest
{
public:
  TTest() : x(-1) {}
  const int& X() const {return x;}
private:
  int x;
  template<typename T> friend void Func(TTest &test,const T &x);
};

template<typename T> void Func(TTest &test,const T &x)
{
  test.x= x;
}

// template<> void Func(TTest &test,const int &x)
// {
  // test.x= 100000;
// }

using namespace std;
#define print(var) std::cout<<#var"= "<<(var)<<std::endl
int main(int argc, char**argv)
{
  TTest test;
  print(test.X());
  Func(test,3.14);
  print(test.X());
  // Func(test,1);
  // print(test.X());
  return 0;
}
//-------------------------------------------------------------------------------------------
