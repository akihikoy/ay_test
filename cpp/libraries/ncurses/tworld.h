//-------------------------------------------------------------------------------------------
/*! \file    tworld.h
    \brief   terminal handling class for games using ncurses
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    Apr.30, 2009
*/
//-------------------------------------------------------------------------------------------
#ifndef tworld_h
#define tworld_h
//-------------------------------------------------------------------------------------------
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <curses.h>
#include <unistd.h>
//-------------------------------------------------------------------------------------------
namespace loco_rabbits
{
//-------------------------------------------------------------------------------------------

//===========================================================================================
class TWorld
//===========================================================================================
{
private:
  int sizex, sizey;
  int oldx, oldy;
  WINDOW *win;
  bool locked;

  struct TCell
    {
      char s;
      int  col;
      TCell(void) : s(' '), col(0) {};
      TCell(char _s) : s(_s), col(0) {};
      TCell(char _s,int _c) : s(_s), col(_c) {};
    };
  std::vector<TCell> display;
  TCell& getCell (int x, int y)  {return display[(sizex+2)*y+x];};
  const TCell& getCell (int x, int y) const {return display[(sizex+2)*y+x];};

public:

  TWorld(void) : sizex(0), sizey(0), oldx(-1), oldy(-1), win(NULL), locked(false) {};
  ~TWorld(void)
    {
      endwin();
    };
  void init(int _sizex, int _sizey, bool hidecursor=true)
    {
      sizex=_sizex; sizey=_sizey;
      if(win!=NULL) {endwin(); win=NULL;}
      display.resize ((sizex+2)*(sizey+2));

      win=initscr();
      timeout(1);
      noecho();
      cbreak();
      leaveok(stdscr, TRUE);
      scrollok(stdscr, FALSE);
      if (hidecursor)  curs_set(0);

      if (has_colors())
      {
        start_color();
        use_default_colors();
        init_pair(0, COLOR_BLACK,   -1);
        init_pair(1, COLOR_RED,     -1);
        init_pair(2, COLOR_GREEN,   -1);
        init_pair(3, COLOR_YELLOW,  -1);
        init_pair(4, COLOR_BLUE,    -1);
        init_pair(5, COLOR_CYAN,    -1);
        init_pair(6, COLOR_MAGENTA, -1);
        init_pair(7, COLOR_WHITE,   -1);
      }

      clear();
      flush();
    };
  void clear(void)
    {
      std::fill (display.begin(),display.end(),TCell());

      // draw walls
      attrset (COLOR_PAIR(2));
      int x,y;
      for(x=1,y=0;x<=sizex;++x)        getCell(x,y)=TCell('-',2);
      for(x=0,y=1;y<=sizey;++y)        getCell(x,y)=TCell('|',2);
      for(x=1,y=sizey+1;x<=sizex;++x)  getCell(x,y)=TCell('-',2);
      for(x=sizex+1,y=1;y<=sizey;++y)  getCell(x,y)=TCell('|',2);
      getCell(0,0)=TCell('+',2);
      getCell(0,sizey+1)=TCell('+',2);
      getCell(sizex+1,0)=TCell('+',2);
      getCell(sizex+1,sizey+1)=TCell('+',2);
    };
  void forceRange(int &x, int &y)
    {
      if(x<0) x=0;
      else if(x>=sizex) x=sizex-1;
      if(y<0) y=0;
      else if(y>=sizey) y=sizey-1;
    };
  void flush (void)
    {
      locked= true;
      std::vector<TCell>::const_iterator itr(display.begin());
      for(int y=0;y<sizey+2;++y)
        for(int x=0;x<sizex+2;++x,++itr)
        {
          attrset (COLOR_PAIR(itr->col));
          mvaddch (y,x,itr->s);
        }
      attrset (COLOR_PAIR(0));
      refresh();
      locked= false;
    };
  void putChar (int x, int y, char s, int col=0)
    {
      forceRange(x,y); ++x; ++y;
      getCell(x,y)=TCell(s,col);
    };
  int getChar (void)
    {
      while (locked) usleep(10);
      return getch();
    };
  void putString (int x, int y, const char *str, int col=0)
    {
      locked= true;
      attrset (COLOR_PAIR(col));
      for (const char *s=str; *s!='\0'; ++s,++x)
        mvaddch(sizey+2+y,x,*s);
      flush();
      locked= false;
    };
};
//-------------------------------------------------------------------------------------------

/* these names conflict with STL */
#undef clear
//-------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------
}
//-------------------------------------------------------------------------------------------
#endif // tworld_h
//-------------------------------------------------------------------------------------------
