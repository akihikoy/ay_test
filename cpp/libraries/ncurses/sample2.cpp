//-------------------------------------------------------------------------------------------
/*! \file    sample2.cpp
    \brief   shooting game sample
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    Apr.01, 2009
    \version 0.2
    \date    Apr.28, 2009  スレッド関連のバグ修正
    \version 0.2
    \date    Apr.30, 2009  ライブラリ化
*/
//-------------------------------------------------------------------------------------------
#include "tworld.h"
#include <sstream>
#include <iomanip>
#include <pthread.h>
#include <cstdlib>
#include <cmath>
//-------------------------------------------------------------------------------------------
inline double u_rand (const double &max)
{
  return (max)*static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
}
inline double u_rand (const double &min, const double &max)
{
  return u_rand(max - min) + min;
}
inline int u_rand (int min, int max)
  // return [min,max]
{
  return floor(u_rand(static_cast<double>(min),static_cast<double>(max+1)));
}
//-------------------------------------------------------------------------------------------
using namespace std;
using namespace loco_rabbits;
//-------------------------------------------------------------------------------------------

struct TMissile
{
  bool active;
  float ry;
  int x,y;
  TMissile(void) : active(false) {}
  void fire (int _x, int _y)
    {
      active= true;
      x= _x;
      y= _y;
      ry= y;
    }
  void step(void)
    {
      ry-=0.5;
      y=static_cast<int>(ry);
    }
};

struct TEnemy
{
  bool active, fired;
  int x,y;
  TEnemy(void) : active(true), fired(false) {}
  void step(void)
    {
      x+= u_rand(-1,1);
      y+= 1;
    }
};

enum TKeyFlag {kfNone=0,kfEnd,kfFire};

class TSimulator
{
private:
  TWorld world;
  int sizex, sizey;
  int x, y;
  int velx, vely;
  bool fired;
  vector<TMissile> missile;
  vector<TEnemy>   enemy;
  TKeyFlag keyflag;
  pthread_t thread_kbhit;
  void step_simulate (void);
  friend void* kbhit (void *sim);  // この関数は private メンバにアクセスできる
public:
  TSimulator (int num_of_enemy=3, int num_of_missile=3)
      : missile(num_of_missile),
        enemy(num_of_enemy)
    {}
  void start (int _sizex, int _sizey);
};
//-------------------------------------------------------------------------------------------

void* kbhit (void *arg)
{
  TSimulator *sim(reinterpret_cast<TSimulator*>(arg));
  while(!sim->fired && sim->keyflag!=kfEnd)
  {
    int key= sim->world.getChar();
    if (key=='q')
    {
      sim->keyflag= kfEnd;
      break;
    }
    if      (key==0x41) sim->vely-=1;
    else if (key==0x42) sim->vely+=1;
    else if (key==0x43) sim->velx+=1;
    else if (key==0x44) sim->velx-=1;
    else if (key==' ')  sim->keyflag= kfFire;
  }
  return NULL;
}
//-------------------------------------------------------------------------------------------

void TSimulator::start (int _sizex, int _sizey)
{
  sizex= _sizex;
  sizey= _sizey;
  world.init (sizex,sizey);
  world.putString(0,0,"usage: arrow key: move, space: missile, `q': exit",2);
  // setup myself
  x= sizex/2;
  y= sizey-1;
  velx=vely= 0;
  fired=false;
  // setup enemy
  for (size_t i(0); i<enemy.size(); ++i)
  {
    enemy[i].x= u_rand(0,sizex-1);
    enemy[i].y= u_rand(0,4);
  }
  // setup missile
  keyflag= kfNone;

  if (pthread_create(&thread_kbhit, NULL, &kbhit, this)!=0)
  {
    cerr<<"error in pthread_create"<<endl;
    exit(1);
  }
  while (keyflag!=kfEnd && !fired)
  {
    world.clear();
    // simulate
    step_simulate();
    // plot
    for (size_t i(0); i<enemy.size(); ++i)
    {
      if (enemy[i].active)
      {
        if (enemy[i].fired) world.putChar (enemy[i].x,enemy[i].y,'*',1);
        else                world.putChar (enemy[i].x,enemy[i].y,'V',6);
      }
    }
    for (size_t i(0); i<missile.size(); ++i)
      if (missile[i].active) world.putChar (missile[i].x,missile[i].y,'^',3);
    if (fired)
    {
      world.putChar (x,y,'*',1);
      world.putString(0,1,"crash! (hit any key)",1);
    }
    else
      world.putChar (x,y,'A',4);
    world.flush();
    usleep(70000);
  }
  usleep(1000000);
  void *ret(NULL);
  pthread_join(thread_kbhit,&ret);
}
//-------------------------------------------------------------------------------------------

void TSimulator::step_simulate (void)
{
  if (keyflag==kfFire)
  {
    keyflag= kfNone;
    for (size_t i(0); i<missile.size(); ++i)
      if (!missile[i].active)  // 使用していないミサイル
        {missile[i].fire(x,y); break;}
  }
  x+= velx;
  y+= vely;
  velx= 0;
  vely= 0;
  world.forceRange(x,y);

  bool no_enemy(true);
  for (size_t i(0); i<enemy.size(); ++i)
  {
    if (enemy[i].fired)  enemy[i].active= false;
    if (enemy[i].active)
    {
      no_enemy= false;
      enemy[i].step();
      if (enemy[i].y>=sizey) enemy[i].y=0;
      world.forceRange(enemy[i].x,enemy[i].y);
      if (enemy[i].x==x && enemy[i].y==y)  fired= true;
    }
  }
  if (no_enemy)
  {
    keyflag= kfEnd;
    world.putString(0,1,"you win! (hit any key)",4);
  }

  for (size_t i(0); i<missile.size(); ++i)
  {
    if (missile[i].active)
    {
      missile[i].step();
      if (missile[i].y<0)  missile[i].active= false;

      for (size_t i(0); i<enemy.size(); ++i)
      {
        if (enemy[i].x==missile[i].x && enemy[i].y==missile[i].y)
        {
          enemy[i].fired= true;
          missile[i].active= false;
        }
      }
    }
  }
}
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  srand((unsigned)time(NULL));
  TSimulator sim;
  sim.start(40,20);
  return 0;
}
//-------------------------------------------------------------------------------------------
