/*
compile: x++ -ode 1-gplanningse3.cpp -- -I ~/prg/ompl/app/ompl/src -I ~/prg/ompl/app/src -L ~/prg/ompl/app/build/lib -lompl -Wl,-rpath ~/prg/ompl/app/build/lib -lboost_thread
exec: ./a.out
*/

// #include <ompl/geometric/SimpleSetup.h>
#include <omplapp/apps/SE3RigidBodyPlanning.h>
#include <ompl/config.h>

namespace ob = ompl::base;
namespace og = ompl::geometric;

bool isStateValid(const ob::State *state)
{
  return true;
}

void planWithSimpleSetup(void)
{
  // construct the state space we are planning in
  ob::StateSpacePtr space(new ob::SE3StateSpace());

  ob::RealVectorBounds bounds(3);
  bounds.setLow(-1);
  bounds.setHigh(1);
  space->as<ob::SE3StateSpace>()->setBounds(bounds);

  og::SimpleSetup ss(space);

  ss.setStateValidityChecker(boost::bind(&isStateValid, _1));

  ob::ScopedState<> start(space);
  start.random();
  std::cout << "start: "; start.print(std::cout);

  ob::ScopedState<> goal(space);
  goal.random();
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
  }

}

int main()
{
  planWithSimpleSetup();
  return 0;
}
