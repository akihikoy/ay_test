/*
OMPL for a continuous 3d-rigit body path planning assuming a UAV
  where the UAV cannot rotate more than 0.X rad in roll/pitch
  this constraint is satisfied by using a valid state sampler
compile: x++ p3-02b-uav.cpp -- -I ~/prg/ompl/app/ompl/src -I ~/prg/ompl/app/src -L ~/prg/ompl/app/build/lib -lompl -Wl,-rpath ~/prg/ompl/app/build/lib -lboost_thread
exec: ./a.out
gnuplot -persistent res/result.gp
gnuplot -persistent res/result.gp > out-p3-02b-uav.svg
*/

#include <ompl/geometric/SimpleSetup.h>
#include <ompl/base/spaces/SE3StateSpace.h>
#include <ompl/geometric/planners/prm/PRM.h>
#include "p3-uav-helper.h"

static const double RobotRot(0.004);

class TMyValidStateSampler : public ob::ValidStateSampler
{
public:
  TMyValidStateSampler(const ob::SpaceInformation *si) : ValidStateSampler(si)
  {
    name_ = "my sampler";
  }
  virtual bool sample(ob::State *state)
  {
    ob::SE3StateSpace::StateType *state_3d= state->as<ob::SE3StateSpace::StateType>();
    // ob::SE3StateSpace::StateType *state_3d= static_cast<ob::SE3StateSpace::StateType*>(state);
    const ob::RealVectorBounds &bounds= si_->getStateSpace()->as<ob::SE3StateSpace>()->getBounds();
    state_3d->setX(rng_.uniformReal(bounds.low[0],bounds.high[0]));
    state_3d->setY(rng_.uniformReal(bounds.low[1],bounds.high[1]));
    state_3d->setZ(rng_.uniformReal(bounds.low[2],bounds.high[2]));
    state_3d->rotation().setIdentity();
    return si_->isValid(state);
  }
  // We don't need this in the example below.
  virtual bool sampleNear(ob::State*, const ob::State*, const double)
  {
    throw ompl::Exception("TMyValidStateSampler::sampleNear", "not implemented");
    return false;
  }
protected:
  ompl::RNG rng_;
};
ob::ValidStateSamplerPtr AllocTMyValidStateSampler(const ob::SpaceInformation *si)
{
  return ob::ValidStateSamplerPtr(new TMyValidStateSampler(si));
}

bool IsStateValid(const ob::State *state)
{
  using namespace std;
  const ob::SE3StateSpace::StateType *state_3d= state->as<ob::SE3StateSpace::StateType>();
  const double &x(state_3d->getX()), &y(state_3d->getY()), &z(state_3d->getZ());
  if(x<0.0 || SizeX<x || y<0.0 || SizeY<y || z<0.0 || SizeZ<z)  return false;

  TMatrix R= QtoR(state_3d->rotation());
  double theta= atan2(sqrt(Sq(R(0,2))+Sq(R(1,2))),R(2,2));
  if(theta>RobotRot)  return false;

  for(vector<TVector>::const_iterator itr(Obstacles.begin()),last(Obstacles.end()); itr!=last; ++itr)
  {
    const double &ox((*itr)(0)), &oy((*itr)(1)), &oz((*itr)(2));
    if(sqrt(Sq(ox-x)+Sq(oy-y)+Sq(oz-z)) < ObstacleRadius+RobotRadius)
      return false;
  }
  return true;
}

void PlanWithSimpleSetup(void)
{
  // construct the state space we are planning in
  ob::StateSpacePtr space(new ob::SE3StateSpace());

  ob::RealVectorBounds bounds(3);
  bounds.setLow(0,0.0);
  bounds.setHigh(0,SizeX);
  bounds.setLow(1,0.0);
  bounds.setHigh(1,SizeY);
  bounds.setLow(2,0.0);
  bounds.setHigh(2,SizeZ);
  space->as<ob::SE3StateSpace>()->setBounds(bounds);

  og::SimpleSetup ss(space);

  ss.getSpaceInformation()->setValidStateSamplerAllocator(AllocTMyValidStateSampler);

  ss.setStateValidityChecker(boost::bind(&IsStateValid, _1));

  ob::ScopedState<ob::SE3StateSpace> start(space);
  start->setXYZ(0.25,0.25,0.25);
  start->rotation().setIdentity();
  std::cout << "start: "; start.print(std::cout);

  ob::ScopedState<ob::SE3StateSpace> goal(space);
  goal->setXYZ(SizeX-0.25,SizeY-0.25,SizeZ-0.25);
  goal->rotation().setIdentity();
  std::cout << "goal: "; goal.print(std::cout);

  ss.setStartAndGoalStates(start, goal);

  // specify the planner
  ob::PlannerPtr planner(new og::PRM(ss.getSpaceInformation()));
  ss.setPlanner(planner);

  std::cout << "----------------" << std::endl;
  ob::PlannerStatus solved = ss.solve(5.0);

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
    // PrintBoxSequence("res/frame_all.dat", ss.getSolutionPath(), RobotX, RobotY, RobotZ);
    PrintSolution("res/result.gp", ss.getSolutionPath());
  }
}

int main()
{
  CreateMap(50, 0);
  // PrintMap();
  PlanWithSimpleSetup();
  return 0;
}
