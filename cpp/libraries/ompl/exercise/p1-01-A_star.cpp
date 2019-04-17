// A* for a discrete maze problem
// x++ p1-01-A_star.cpp
// qplot -s 'set size ratio 0.83' map.dat pt 5 ps 9 t '""' frame00000.dat w l lt 3 lw 5 t '""'
// qplot -s 'set size ratio 0.83' map.dat pt 5 ps 5 t '""' frame00000.dat w l lt 3 lw 5 t '""' -o 'out-p1-01-A_star.svg'


#include "p1-01-A_star.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cmath>
#include <unistd.h>
using namespace std;

static const int W(6),H(5);
// static const int W(20),H(20);

void StateToXY(const TAStarSearch::TState &state, int &x, int &y)
{
  x= state%W;
  y= (state-x)/W;
}
TAStarSearch::TState XYToState(int x, int y)
{
  if(x<0 || W<=x || y<0 || H<=y)  return TAStarSearch::InvalidState;
  return y*W+x;
}

TAStarSearch::TValue CostFunction(const TAStarSearch::TState &state1, const TAStarSearch::TState &state2)
{
  return 0.1l;
}
TAStarSearch::TValue HeuristicCostFunction(const TAStarSearch::TState &state1, const TAStarSearch::TState &state2)
{
  int x1,y1,x2,y2;
  StateToXY(state1,x1,y1);
  StateToXY(state2,x2,y2);
  return 0.1l*(std::fabs(x2-x1)+std::fabs(y2-y1));
  // return 0.1l*std::sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1));
  // return 0.0;
}

bool StateValidity(const TAStarSearch::TState &state)
{
  if(state==TAStarSearch::InvalidState)  return false;
  int walls[]= {1,7,13, 16,22,28};
  for(size_t i(0); i<sizeof(walls)/sizeof(walls[0]); ++i)
    if(state==walls[i])  return false;
  return true;
}

std::list<TAStarSearch::TState> SuccessorsFunction(const TAStarSearch::TState &state)
{
  std::list<TAStarSearch::TState> successors;
  int x,y;
  StateToXY(state,x,y);
  TAStarSearch::TState n;
  if(StateValidity(n=XYToState(x+1,y)))   successors.push_back(n);
  if(StateValidity(n=XYToState(x-1,y)))   successors.push_back(n);
  if(StateValidity(n=XYToState(x,y+1)))   successors.push_back(n);
  if(StateValidity(n=XYToState(x,y-1)))   successors.push_back(n);
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
      int x1,y1,x2,y2;
      StateToXY(s1,x1,y1);
      StateToXY(s2,x2,y2);
      ofs<<endl;
      ofs<<x1<<" "<<y1<<endl;
      ofs<<x2<<" "<<y2<<endl;
    }
  }
  usleep(500*1000);
}

void PrintMap()
{
  ofstream ofs("res/map.dat");
  for(int x(-1); x<=W; ++x)
  {
    for(int y(-1); y<=H; ++y)
    {
      if(!StateValidity(XYToState(x,y)))
        ofs<<x<<" "<<y<<endl;
    }
  }
}

int main(int argc,char**argv)
{
  PrintMap();

  TAStarSearch a_star;

  a_star.SetStateMax(W*H-1);
  a_star.SetProblem(0, 29);
  // a_star.SetProblem(0, 399);

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

