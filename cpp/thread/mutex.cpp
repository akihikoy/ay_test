//-------------------------------------------------------------------------------------------
/*! \file    boost-th.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    Oct.29, 2012

g++ -g -Wall mutex.cpp -lboost_system -lboost_thread -Wl,-rpath /usr/lib/x86_64-linux-gnu/
*/
//-------------------------------------------------------------------------------------------
#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include <iostream>
#include <unistd.h>
// #include <lora/util.h>
// #include <lora/common.h>
// #include <lora/math.h>
// #include <lora/file.h>
// #include <lora/rand.h>
// #include <lora/small_classes.h>
// #include <lora/stl_ext.h>
// #include <lora/string.h>
// #include <lora/sys.h>
// #include <lora/octave.h>
// #include <lora/octave_str.h>
// #include <lora/ctrl_tools.h>
// #include <lora/ode.h>
// #include <lora/ode_ds.h>
// #include <lora/vector_wrapper.h>
// #include <iostream>
// #include <iomanip>
// #include <string>
// #include <vector>
// #include <list>
// #include <boost/lexical_cast.hpp>
//-------------------------------------------------------------------------------------------
namespace loco_rabbits
{

struct TTest
{
  int Sum;
  bool Flag,IsExec;
  boost::thread *th;
  boost::mutex mutex_;

  TTest() : th(NULL)
    {
      Sum= 0;
      Flag= true;
      IsExec= false;
      th= new boost::thread(boost::bind(&TTest::Exec,this));
    }
  ~TTest()
    {
      Flag= false;
      std::cout<<"stopping..."<<std::endl;
      std::cout<<std::flush;
      if(th)
      {
        th->join();
        delete th;
      }
      th= NULL;
      std::cout<<"stopped."<<std::endl;
      std::cout<<std::flush;
    }

  void Exec()
    {
      std::cout<<"starting..."<<std::endl;
      std::cout<<std::flush;
      while(Flag)
      {
        {
          boost::mutex::scoped_lock lock(mutex_);
          for(int j(0);j<1000;++j)
          {
            Sum+=1;
          }
          std::cout<<"TTest: "<<Sum<<std::endl;
          std::cout<<std::flush;
        }
        usleep(10000);
      }
    }
  int Get()
    {
      boost::mutex::scoped_lock lock(mutex_);
      return Sum;
    }
};

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
  TTest test;
  int A(0);
  for(int i(0);i<100;++i)
  {
    for(int j(0);j<500;++j)
    {
      A+=1;
    }
    std::cout<<"main: "<<A<<" : "<<test.Get()<<std::endl;
    std::cout<<std::flush;
    usleep(5000);
  }
  return 0;
}
//-------------------------------------------------------------------------------------------
