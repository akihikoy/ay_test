//-------------------------------------------------------------------------------------------
/*! \file    kbhit-test.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    Dec.03, 2013

    compile:
      x++ -slora kbhit-test.cpp

    usage:
      ./a.out
*/
//-------------------------------------------------------------------------------------------
#include <lora/sys.h>
// #include <iostream>
// #include <iomanip>
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
  cout<<"hit space > "<<flush;
  WaitKBHit(' ');
  cout<<"ok."<<endl;
  cout<<"hit any key > "<<flush;
  char s= WaitKBHit();
  cout<<"ok.(\'"<<s<<"\' has been hit)"<<endl;
  {
    cout<<"no-wait mode using KBHit(). hit any key to stop.."<<endl;
    while(true)
    {
      cout<<"+"<<flush;
      int c= KBHit();
      if(c=='q')  break;
    }
    cout<<"stopped"<<endl;
  }
  {
    cout<<"no-wait mode using TKBHit. hit any key to stop.."<<endl;
    TKBHit kbhit(false); // no-wait
    while(true)
    {
      cout<<"+"<<flush;
      int c= kbhit();
      if(c=='q')  break;
    }
  }
  cout<<"stopped"<<endl;

  return 0;
}
//-------------------------------------------------------------------------------------------
