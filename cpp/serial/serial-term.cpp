//-------------------------------------------------------------------------------------------
/*! \file    serial-term.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    Oct.18, 2013

    x++ -slora serial-term.cpp

    with bluetooth:
    (in another terminal)
      hcitool scan
      rfcomm connect -i 00:09:E7:02:37:2A
      # Connected /dev/rfcomm0 to ...
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
#include <lora/serial.h>
#include <lora/sys.h>  // TKBHit
// #include <lora/octave.h>
// #include <lora/octave_str.h>
// #include <lora/ctrl_tools.h>
// #include <lora/ode.h>
// #include <lora/ode_ds.h>
// #include <lora/vector_wrapper.h>
// #include <iostream>
// #include <iomanip>
#include <string>
#include <cstring>
// #include <vector>
// #include <list>
// #include <boost/lexical_cast.hpp>
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

//-------------------------------------------------------------------------------------------
const int    BUFFER_SIZE(2048);
//-------------------------------------------------------------------------------------------

termios GetDefaultTermios (void)
{
  termios ios;
  memset(&ios, 0, sizeof(ios));
  // ios.c_cflag |= B115200 | CREAD | CLOCAL | CS8;  /* control mode flags */
  ios.c_cflag |= B57600 | CREAD | CLOCAL | CS8;  /* control mode flags */
  ios.c_iflag |= IGNPAR;  /* input mode flags */
  ios.c_cc[VMIN] = 0;
  ios.c_cc[VTIME] = 1;  // *[1/10 sec] for timeout
  return ios;
}

int main(int argc, char**argv)
{
  serial::TSerialCom  serial_;
  serial::TStringWriteStream   str_writes_;
  unsigned char buffer_[BUFFER_SIZE];
  // int N(BUFFER_SIZE);

  // setup configuration
  // std::string v_tty("/dev/ttyUSB0");
  std::string v_tty("/dev/rfcomm0");
  termios v_ios(GetDefaultTermios());

  // setup connection
  if(serial_.IsOpen())  serial_.Close();
  serial_.setting(v_tty,v_ios);
  serial_.Open();

  if(!serial_.IsOpen())
  {
    LERROR("failed to connect: "<<v_tty);
    lexit(df);
  }

  cout<<"select c:command by command, i:character by character (c|i) > "<<flush;
  char c;
  cin>>c;
  if(c=='c')
  {
    while(true)
    {
      string command;
      cout<<" > "<<flush;
      cin>>command;
      if(command=="exit")  break;

      cout<<"write: "<<command<<"("<<command.length()<<")"<<endl;
      str_writes_<<command.c_str();
      str_writes_>>serial_;

      int n= serial_.Read(buffer_, BUFFER_SIZE-1);
      buffer_[n]='\0';
      cout<<"read("<<n<<"): "<<buffer_<<endl;
    }
  }
  else if(c=='i')
  {
    cout<<"type a key for exit > "<<flush;
    char x_command, command[2]={0,0};
    cin>>x_command;
    TKBHit kbhit(false);
    while(true)
    {
      command[0]= kbhit();
      if(command[0]==x_command)
        break;
      else if(command[0]!=0)
      {
        // cout<<"write: "<<command[0]<<"(1)"<<endl;
        str_writes_<<command;
        str_writes_>>serial_;
      }

      int n= serial_.Read(buffer_, BUFFER_SIZE-1);
      buffer_[n]='\0';
      // cout<<"read("<<n<<"): "<<buffer_<<endl;
      if(n>0)  cout<<buffer_<<flush;
    }
    kbhit.Close();
    cerr<<endl;
  }

  serial_.Close();

  return 0;
}
//-------------------------------------------------------------------------------------------
