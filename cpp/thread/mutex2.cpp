//-------------------------------------------------------------------------------------------
/*! \file    mutex2.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    May.10, 2016

g++ -g -Wall mutex2.cpp -lboost_system -lboost_thread -Wl,-rpath /usr/lib/x86_64-linux-gnu/
*/
//-------------------------------------------------------------------------------------------
#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include <iostream>
#include <unistd.h>
//-------------------------------------------------------------------------------------------
namespace loco_rabbits
{

int X1(0), X2(0);

boost::mutex Mutex1, Mutex2;
bool Running(true);

void Print1()
{
  int count(0);
  while(Running)
  {
    {
      boost::mutex::scoped_lock lock(Mutex1);
      std::cout<<"Print1: "<<X1<<"..."<<std::endl;
      usleep(1000*1000);
      std::cout<<"Print1: done."<<std::endl;
    }
    // usleep(2000*1000);
    usleep(1*1000);
    ++count;
  }
  std::cout<<"Print1: looped "<<count<<std::endl;
}

void Print2()
{
  int count(0);
  while(Running)
  {
    {
      boost::mutex::scoped_lock lock(Mutex2);
      std::cout<<"Print2: "<<X2<<"..."<<std::endl;
      usleep(1500*1000);
      std::cout<<"Print2: done."<<std::endl;
    }
    // usleep(1000*1000);
    usleep(1*1000);
    ++count;
  }
  std::cout<<"Print2: looped "<<count<<std::endl;
}


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
  boost::thread th1(&Print1);
  boost::thread th2(&Print2);
  for(int i(0);i<5;++i)
  {
    {
      boost::mutex::scoped_lock lock(Mutex1);
      X1+= 1;
      std::cout<<"Main-X1: changed: "<<X1<<"..."<<std::endl;
      usleep(2000*1000);
      std::cout<<"Main-X1: done."<<std::endl;
    }
    {
      boost::mutex::scoped_lock lock(Mutex2);
      X2+= 10;
      std::cout<<"Main-X2: changed: "<<X2<<"..."<<std::endl;
      usleep(2000*1000);
      std::cout<<"Main-X2: done."<<std::endl;
    }
    // usleep(5000*1000);
  }
  Running= false;
  th1.join();
  th2.join();
  return 0;
}
//-------------------------------------------------------------------------------------------
