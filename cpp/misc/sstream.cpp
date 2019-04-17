//-------------------------------------------------------------------------------------------
/*! \file    ifstream.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    Aug.02, 2011
*/
//-------------------------------------------------------------------------------------------
#include <iostream>
#include <sstream>
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
  string line("10 20 30 40");
  {
    stringstream ss(line);
    double value;
    for(int i(0);i<10;++i)
    {
      print(ss.str());
      ss>>value;
      print(ss.str());
      print(value);
    }
  }
  cout<<"---------------------"<<endl;
  {
    stringstream ss(line);
    double value;
    while(ss>>value)
    {
      print(ss.str());
      print(value);
    }
  }
  return 0;
}
//-------------------------------------------------------------------------------------------
