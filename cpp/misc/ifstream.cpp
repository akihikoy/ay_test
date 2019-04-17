//-------------------------------------------------------------------------------------------
/*! \file    ifstream.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    Aug.02, 2011
*/
//-------------------------------------------------------------------------------------------
#include <iostream>
#include <fstream>
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
// #define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  if(argc!=2)  {cerr<<"specify a FILE."<<endl; return 1;}
  ifstream ifs(argv[1]);
  string line;
  int i(1);
  // while(getline(ifs,line))
  while(getline(ifs,line,'\n'))
  {
    cout<<"LINE:"<<i<<": >|| "<<line<<" ||<"<<endl;
    ++i;
  }
  return 0;
}
//-------------------------------------------------------------------------------------------
