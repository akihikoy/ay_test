//-------------------------------------------------------------------------------------------
/*! \file    typedef_array.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Apr.08, 2016
*/
//-------------------------------------------------------------------------------------------
#include <list>
#include <iostream>
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
  typedef float TXY[2];
  list<TXY> xy;
  for(int i(0);i<10;++i)
  {
    TXY p={float(i),float(i*i)};
    xy.push_back(p);
  }
  for(list<TXY>::iterator itr(xy.begin()),last(xy.end()); itr!=last; ++itr)
    cout<<(*itr)[0]<<" "<<(*itr)[1]<<endl;
  return 0;
}
//-------------------------------------------------------------------------------------------
