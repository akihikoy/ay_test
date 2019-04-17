//-------------------------------------------------------------------------------------------
/*! \file    vector_cap.cpp
    \brief   How does vector.capacity change?
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Apr.20, 2016
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
#define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  std::vector<int> x(3,0);
  cout<<"size,capacity= "<<x.size()<<", "<<x.capacity()<<endl;
  x.push_back(1);
  x.push_back(1);
  cout<<"size,capacity= "<<x.size()<<", "<<x.capacity()<<endl;
  x.clear();
  cout<<"size,capacity= "<<x.size()<<", "<<x.capacity()<<endl;
  x.resize(2,0);
  cout<<"size,capacity= "<<x.size()<<", "<<x.capacity()<<endl;
  x.push_back(2);
  cout<<"size,capacity= "<<x.size()<<", "<<x.capacity()<<endl;
  x.resize(0);
  cout<<"size,capacity= "<<x.size()<<", "<<x.capacity()<<endl;
  return 0;
}
//-------------------------------------------------------------------------------------------
