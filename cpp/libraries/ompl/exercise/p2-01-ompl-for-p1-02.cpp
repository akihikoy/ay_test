/*
OMPL for a continuous 2d path planning
compile: x++ p2-01-ompl-for-p1-02.cpp -- -I ~/prg/ompl/app/ompl/src -I ~/prg/ompl/app/src -L ~/prg/ompl/app/build/lib -lompl -Wl,-rpath ~/prg/ompl/app/build/lib -lboost_thread
exec: ./a.out
qplot -s 'set size ratio 0.857; set xrange [-0.5:6.5]; set yrange [-0.5:5.5]' map.dat pt 7 ps 0.1 t '""' frame00000.dat w l lt 2 lw 1 t '"A*"' path.dat w l lt 3 lw 3 t '"LBKPIECE1"'
qplot -s 'set size ratio 0.857; set xrange [-0.5:6.5]; set yrange [-0.5:5.5]' map.dat pt 7 ps 0.1 t '""' frame00000.dat w l lt 2 lw 1 t '"A*"' path.dat w l lt 3 lw 3 t '"LBKPIECE1"' -o out-p2-01-ompl-for-p1-02.svg
*/

#include <ompl/geometric/SimpleSetup.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <iostream>
#include <fstream>

namespace ob = ompl::base;
namespace og = ompl::geometric;

static const double W(6),H(5);
static const double CellX(0.5), CellY(0.5);
static const double SampleRatio(0.05);

bool StateValidity(const double &x, const double &y)
{
  if(x<0.0 || W<x || y<0.0 || H<y)  return false;
  if(y<4.5*x-2.25 && y<-4.5*x+11.25)  return false;
  if(y<4.4*x-12.9 && y<-4.4*x+22.3 && y>-4.0*x+16.5 && y>4.0*x-15.5)  return false;
  return true;
}

void PrintMap()
{
  using namespace std;
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

bool IsStateValid(const ob::State *state)
{
  const ob::RealVectorStateSpace::StateType *state_2d= state->as<ob::RealVectorStateSpace::StateType>();
  return StateValidity((*state_2d)[0], (*state_2d)[1]);
}

void PlanWithSimpleSetup(void)
{
  // construct the state space we are planning in
  ob::StateSpacePtr space(new ob::RealVectorStateSpace(2));

  ob::RealVectorBounds bounds(2);
  bounds.setLow(0,0.0);
  bounds.setHigh(0,W);
  bounds.setLow(1,0.0);
  bounds.setHigh(1,H);
  space->as<ob::RealVectorStateSpace>()->setBounds(bounds);

  og::SimpleSetup ss(space);

  ss.setStateValidityChecker(boost::bind(&IsStateValid, _1));

  ob::ScopedState<ob::RealVectorStateSpace> start(space);
  start[0]= CellX*0.5;
  start[1]= CellY*0.5;
  std::cout << "start: "; start.print(std::cout);

  ob::ScopedState<ob::RealVectorStateSpace> goal(space);
  goal[0]= W-CellX*0.5;
  goal[1]= H-CellY*0.5;
  std::cout << "goal: "; goal.print(std::cout);

  ss.setStartAndGoalStates(start, goal);

  std::cout << "----------------" << std::endl;
  ob::PlannerStatus solved = ss.solve(1.0);

  if (solved)
  {
    std::cout << "----------------" << std::endl;
    std::cout << "Found solution:" << std::endl;
    // print the path to screen
    ss.simplifySolution();
    ss.getSolutionPath().print(std::cout);
    std::cout << "----------------" << std::endl;
    std::ofstream ofs("res/path.dat");
    ss.getSolutionPath().printAsMatrix(ofs);
  }
}

int main()
{
  PrintMap();
  PlanWithSimpleSetup();
  return 0;
}
