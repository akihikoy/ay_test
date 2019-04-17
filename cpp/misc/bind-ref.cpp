//-------------------------------------------------------------------------------------------
/*! \file    bind-ref.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    Oct.30, 2013
*/
//-------------------------------------------------------------------------------------------
#include <iostream>
// #include <iomanip>
// #include <string>
// #include <vector>
// #include <list>
#include <boost/bind.hpp>
#include <boost/function.hpp>
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
#define pl() std::cout<<__LINE__<<std::endl
#define print(var) std::cout<<"  "#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

void func(const int &x)
{
  print(x);
  print(&x);
}
void set(boost::function<void(void)> &f)
{
  int y= 10;
  pl(); print(&y);
  f= boost::bind(&func,y);
  pl(); func(y);
  pl(); f();
}
void test()
{
  int y= 111;
  pl(); print(&y);
}

int main(int argc, char**argv)
{
  boost::function<void(void)> f;
  set(f);
  pl(); f();
  int z=100;
  pl(); print(&z);
  pl(); f();

  test();
  pl(); f();
  return 0;
}
//-------------------------------------------------------------------------------------------
