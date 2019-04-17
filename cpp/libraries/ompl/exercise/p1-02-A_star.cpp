// A* for a continuous 2d path planning
// x++ p1-02-A_star.cpp
// qplot -s 'set size ratio 0.857; set xrange [-0.5:6.5]; set yrange [-0.5:5.5]' map.dat pt 7 ps 0.1 t '""' frame00000.dat w l lt 3 lw 3 t '""'
// qplot -s 'set size ratio 0.857; set xrange [-0.5:6.5]; set yrange [-0.5:5.5]' map.dat pt 7 ps 0.1 t '""' frame00000.dat w l lt 3 lw 3 t '""' -o out-p1-02-A_star.svg

#include "p1-01-A_star.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cmath>
#include <unistd.h>
using namespace std;

static const double W(6),H(5);
static const double CellX(0.5), CellY(0.5);
static const double SampleRatio(0.05);
static const int NumX(round(W/CellX)), NumY(round(H/CellY));

void StateToXY(const TAStarSearch::TState &state, double &x, double &y)
{
  x= double(state%NumX)*CellX+CellX*0.5;
  y= double((state-state%NumX)/NumX)*CellY+CellY*0.5;
}
TAStarSearch::TState XYToState(const double &x_in, const double &y_in)
{
  if(x_in<0.0 || W<x_in || y_in<0.0 || H<y_in)  return TAStarSearch::InvalidState;
  int x= round((x_in-CellX*0.5)/CellX), y= round((y_in-CellY*0.5)/CellY);
  return y*NumX+x;
}

TAStarSearch::TValue CostFunction(const TAStarSearch::TState &state1, const TAStarSearch::TState &state2)
{
  return 0.1l;
}
TAStarSearch::TValue HeuristicCostFunction(const TAStarSearch::TState &state1, const TAStarSearch::TState &state2)
{
  double x1,y1,x2,y2;
  StateToXY(state1,x1,y1);
  StateToXY(state2,x2,y2);
  // return 0.1l*(std::fabs(x2-x1)+std::fabs(y2-y1));
  return 0.1l*std::sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1));
  // return 0.0;
}

bool StateValidity(const double &x, const double &y)
{
  if(x<0.0 || W<x || y<0.0 || H<y)  return false;
  if(y<4.5*x-2.25 && y<-4.5*x+11.25)  return false;
  if(y<4.4*x-12.9 && y<-4.4*x+22.3 && y>-4.0*x+16.5 && y>4.0*x-15.5)  return false;
  return true;
}

bool DStateValidity(const TAStarSearch::TState &state)
{
  static bool init(true);
  static vector<int>  walls;
  if(init)
  {
    init= false;
    for(int s(0); s<NumX*NumY; ++s)
    {
      bool is_wall= false;
      double cx,cy;
      StateToXY(s,cx,cy);
      for(double x(cx-CellX*0.5); x<cx+CellX*0.5; x+=CellX*SampleRatio)
      {
        for(double y(cy-CellY*0.5); y<cy+CellY*0.5; y+=CellY*SampleRatio)
          if(!StateValidity(x,y))  {is_wall= true; break;}
        if(is_wall)  break;
      }
      if(is_wall)  walls.push_back(s);
    }
    // for(size_t i(0); i<walls.size(); ++i) cerr<<" "<<walls[i]; cerr<<endl;
  }
  if(state==TAStarSearch::InvalidState)  return false;
  for(size_t i(0); i<walls.size(); ++i)
    if(state==walls[i])  return false;
  return true;
}

std::list<TAStarSearch::TState> SuccessorsFunction(const TAStarSearch::TState &state)
{
  std::list<TAStarSearch::TState> successors;
  double x,y;
  StateToXY(state,x,y);
  TAStarSearch::TState n;
  if(DStateValidity(n=XYToState(x+CellX,y)))   successors.push_back(n);
  if(DStateValidity(n=XYToState(x-CellX,y)))   successors.push_back(n);
  if(DStateValidity(n=XYToState(x,y+CellY)))   successors.push_back(n);
  if(DStateValidity(n=XYToState(x,y-CellY)))   successors.push_back(n);
  return successors;
}

void OnUpdated(const TAStarSearch &a, const TAStarSearch::TState &state1, const TAStarSearch::TState &state2)
{
  static int idx(0);
  stringstream ss;
  ss<<"res/frame"<<setfill('0')<<setw(5)<<idx<<".dat";
  // ++idx;
  ofstream ofs(ss.str().c_str());
  const vector<TAStarSearch::TState>& ip(a.InversePolicy());
  for(TAStarSearch::TState s2(0); s2<=a.StateMax(); ++s2)
  {
    if(ip[s2]!=TAStarSearch::InvalidState)
    {
      TAStarSearch::TState s1= ip[s2];
      double x1,y1,x2,y2;
      StateToXY(s1,x1,y1);
      StateToXY(s2,x2,y2);
      ofs<<endl;
      ofs<<x1<<" "<<y1<<endl;
      ofs<<x2<<" "<<y2<<endl;
    }
  }
  usleep(100*1000);
}

void PrintMap()
{
  ofstream ofs("res/map.dat");
  for(double x(-CellX*0.5); x<=W+CellX*0.5; x+=CellX*SampleRatio)
  {
    for(double y(-CellY*0.5); y<=H+CellY*0.5; y+=CellY*SampleRatio)
    {
      if(!StateValidity(x,y))
        ofs<<x<<" "<<y<<endl;
    }
  }
}

int main(int argc,char**argv)
{
  PrintMap();

  TAStarSearch a_star;

  a_star.SetStateMax(NumX*NumY-1);
  a_star.SetProblem(XYToState(CellX*0.5,CellY*0.5), XYToState(W-CellX*0.5,H-CellY*0.5));

  a_star.SetCostFunction(&CostFunction);
  a_star.SetHeuristicCostFunction(&HeuristicCostFunction);
  a_star.SetSuccessorsFunction(&SuccessorsFunction);

  a_star.SetCallbackOnUpdated(&OnUpdated);

  a_star.Solve();

  list<TAStarSearch::TState> path= a_star.GetSolvedPath();

  // const std::vector<TState>& InversePolicy() const {return inverse_policy_;}

  cout<<"shortest path: ";
  for(list<TAStarSearch::TState>::iterator itr(path.begin()),last(path.end()); itr!=last; ++itr)
    cout<<" "<<*itr;
  cout<<endl;

  return 0;
}

