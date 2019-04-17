//-------------------------------------------------------------------------------------------
/*! \file    test.cpp
    \brief   test of TWorld class
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    May.01, 2009

    コンパイル:  g++ test.cpp -lncurses -pthread
*/
//-------------------------------------------------------------------------------------------
#include "tworld.h"
#include <pthread.h>
//-------------------------------------------------------------------------------------------
using namespace std;
using namespace loco_rabbits;
//-------------------------------------------------------------------------------------------

bool running (true);
TWorld world;

struct TRobot
{
  int x,y;
  TRobot(void) : x(0), y(0) {};
  void move(int dx, int dy) {x+=dx; y+=dy;};
};

void* kbhit (void *arg)
{
  TRobot &robot(*reinterpret_cast<TRobot*>(arg));
  while(running)
  {
    int key= world.getChar();
    if (key=='q')
    {
      running= false;
      break;
    }
    if      (key==0x41) robot.move(0,-1);
    else if (key==0x42) robot.move(0,+1);
    else if (key==0x43) robot.move(+1,0);
    else if (key==0x44) robot.move(-1,0);
  }
  return NULL;
}
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  pthread_t thread_kbhit;
  TRobot robot;

  world.init(40,20);

  if (pthread_create(&thread_kbhit, NULL, &kbhit, &robot)!=0)
  {
    cerr<<"error in pthread_create"<<endl;
    exit(1);
  }

  world.putString(0,0,"usage: arrow key: move, `q': exit",2);

  while(running)
  {
    world.clear();
    world.putChar (robot.x, robot.y, 'O', 4);
    world.flush();
    usleep(70000);
  }

  world.putString(0,1,"quit",2);
  usleep(500000);

  void *ret(NULL);
  pthread_join(thread_kbhit,&ret);
  return 0;
}
//-------------------------------------------------------------------------------------------
