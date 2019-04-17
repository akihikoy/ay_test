//-------------------------------------------------------------------------------------------
/*! \file    kbhit.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    Jul.27, 2010
*/
//-------------------------------------------------------------------------------------------
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
#include <iostream>
#include <cstdio>
#include <termios.h>
// #include <string>
// #include <vector>
// #include <list>
// #include <boost/lexical_cast.hpp>
//-------------------------------------------------------------------------------------------
namespace loco_rabbits
{


class TKBHit
{
public:
  TKBHit (void) : is_open_(false)
    {
      Open();
    }
  ~TKBHit (void)
    {
      Close();
    }
  void Open (void)
    {
      if (is_open_)  Close();
      tcgetattr(STDIN_FILENO, &old_tios_);

      raw_tios_= old_tios_;
      cfmakeraw(&raw_tios_);

      tcsetattr(STDIN_FILENO, 0, &raw_tios_);
      is_open_= true;
    }
  void Close (void)
    {
      if (!is_open_)  return;
      tcsetattr(STDIN_FILENO, 0, &old_tios_);
      std::cout<<std::endl;
      is_open_= false;
    }
  int operator() (void) const
    {
      return getchar();
    }
private:
  struct termios old_tios_;
  struct termios raw_tios_;
  bool is_open_;
};
//-------------------------------------------------------------------------------------------


inline int WaitKBHit(void)
{
  TKBHit kbhit;
  return kbhit();
}
inline void WaitKBHit(char k)
{
  TKBHit kbhit;
  while(1)
  {
    int s= kbhit();
    if(s==k)  break;
  }
}

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
  return 0;
}
//-------------------------------------------------------------------------------------------
