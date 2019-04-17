/*
OMPL for a continuous 3d-rigit body path planning assuming a UAV
  where the UAV is constrained by a simplified dynamics
compile: x++ p3-03-uavdyn.cpp -- -I ~/prg/ompl/app/ompl/src -I ~/prg/ompl/app/src -L ~/prg/ompl/app/build/lib -lompl -Wl,-rpath ~/prg/ompl/app/build/lib -lboost_thread
exec: ./a.out
gnuplot -persistent res/result.gp
gnuplot -persistent res/result.gp > out-p3-03-uavdyn.svg
*/

///>>>+++
#include <ompl/control/SimpleSetup.h>
#include <ompl/control/spaces/RealVectorControlSpace.h>
///<<<+++
#include <ompl/base/spaces/SE3StateSpace.h>
#include "p3-uav-helper.h"

namespace oc = ompl::control;

static const double MaxUf(0.3),MaxUh(0.3),MaxUr(0.3);
static const double MinUf(-0.3),MinUh(-0.3),MinUr(-0.3);


bool IsStateValid(const ob::State *state)
{
  using namespace std;
  const ob::SE3StateSpace::StateType *state_3d= state->as<ob::SE3StateSpace::StateType>();
  const double &x(state_3d->getX()), &y(state_3d->getY()), &z(state_3d->getZ());
  if(x<0.0 || SizeX<x || y<0.0 || SizeY<y || z<0.0 || SizeZ<z)  return false;

  for(vector<TVector>::const_iterator itr(Obstacles.begin()),last(Obstacles.end()); itr!=last; ++itr)
  {
    const double &ox((*itr)(0)), &oy((*itr)(1)), &oz((*itr)(2));
    if(sqrt(Sq(ox-x)+Sq(oy-y)+Sq(oz-z)) < ObstacleRadius+RobotRadius)
      return false;
  }
  return true;
}

///>>>+++
void Propagate(const ob::State *start, const oc::Control *control, const double duration, ob::State *result)
{
  const ob::SE3StateSpace::StateType *start_3d= start->as<ob::SE3StateSpace::StateType>();
  const oc::RealVectorControlSpace::ControlType *control_rv= control->as<oc::RealVectorControlSpace::ControlType>();
  ob::SE3StateSpace::StateType *result_3d= result->as<ob::SE3StateSpace::StateType>();

  TMatrix R= QtoR(start_3d->rotation());

  result_3d->setXYZ(
      start_3d->getX() + duration*(*control_rv)[0]*R(0,0) + duration*(*control_rv)[1]*R(0,2),
      start_3d->getY() + duration*(*control_rv)[0]*R(1,0) + duration*(*control_rv)[1]*R(1,2),
      start_3d->getZ() + duration*(*control_rv)[0]*R(2,0) + duration*(*control_rv)[1]*R(2,2)
    );

  ob::SO3StateSpace::StateType rot1;
  rot1.setAxisAngle(R(0,2),R(1,2),R(2,2), duration*(*control_rv)[2]);
  QuaternionProduct(result_3d->rotation(),/*=*/rot1,/***/start_3d->rotation());
}
///<<<+++

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

///>>>+++
  // construct the control space
  oc::ControlSpacePtr cspace(new oc::RealVectorControlSpace(space, 3));

  ob::RealVectorBounds cbounds(3);
  cbounds.setLow(0,MinUf);
  cbounds.setHigh(0,MaxUf);
  cbounds.setLow(1,MinUh);
  cbounds.setHigh(1,MaxUh);
  cbounds.setLow(2,MinUr);
  cbounds.setHigh(2,MaxUr);
  cspace->as<oc::RealVectorControlSpace>()->setBounds(cbounds);
///<<<+++

///>>>CHANGE
  oc::SimpleSetup ss(cspace);
///<<<CHANGE

  ss.setStateValidityChecker(boost::bind(&IsStateValid, _1));

///>>>+++
  ss.setStatePropagator(boost::bind(&Propagate, _1, _2, _3, _4));
///<<<+++

  ob::ScopedState<ob::SE3StateSpace> start(space);
  start->setXYZ(0.25,0.25,0.25);
  start->rotation().setIdentity();
  std::cout << "start: "; start.print(std::cout);

  ob::ScopedState<ob::SE3StateSpace> goal(space);
  goal->setXYZ(SizeX-0.25,SizeY-0.25,SizeZ-0.25);
  goal->rotation().setIdentity();
  std::cout << "goal: "; goal.print(std::cout);

///>>>CHANGE
  ss.setStartAndGoalStates(start, goal, 0.2);
///<<<CHANGE

///>>>+++
  ss.getSpaceInformation()->setMinMaxControlDuration(1,200);
  ss.getSpaceInformation()->setPropagationStepSize(0.05);
///<<<+++

  std::cout << "----------------" << std::endl;
  ob::PlannerStatus solved = ss.solve(5.0);

  if (solved)
  {
    std::cout << "----------------" << std::endl;
    std::cout << "Found solution:" << std::endl;
///>>>---
    // ss.simplifySolution();
///<<<---
    // print the path to screen
    ss.getSolutionPath().print(std::cout);
    std::cout << "----------------" << std::endl;
    std::ofstream ofs("res/path.dat");
    ss.getSolutionPath().printAsMatrix(ofs);
    // PrintBoxSequence("res/frame_all.dat", ss.getSolutionPath(), RobotX, RobotY, RobotZ);
    PrintSolution("res/result.gp", ss.getSolutionPath().asGeometric(), 50);
  }
}

int main()
{
  CreateMap(50, 0);
  // PrintMap();
  PlanWithSimpleSetup();
  return 0;
}
