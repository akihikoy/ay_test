//-------------------------------------------------------------------------------------------
/*! \file    same-name-inherit.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    Feb.08, 2012
*/
//-------------------------------------------------------------------------------------------
#include <iostream>
using namespace std;
namespace nsa
{
  class TTest
  {
  public:
    int X;
    virtual void print() const {cout<<"X: "<<X<<endl;}
  };
}
namespace nsb
{
  class TTest : public nsa::TTest
  {
  public:
    int Y;
    /*override*/void print() const
      {
        nsa::TTest::print();
        cout<<"Y: "<<Y<<endl;
      }
  };
}
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  nsb::TTest test;
  test.X= 10;
  test.Y= -1;
  test.print();
  return 0;
}
//-------------------------------------------------------------------------------------------
