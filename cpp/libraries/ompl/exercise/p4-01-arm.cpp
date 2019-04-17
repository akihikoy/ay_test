/*
OMPL for a continuous 2d path planning
compile: x++ p4-01-arm.cpp -- -I ~/prg/ompl/app/ompl/src -I ~/prg/ompl/app/src -L ~/prg/ompl/app/build/lib -lompl -Wl,-rpath ~/prg/ompl/app/build/lib -lboost_thread
exec: ./a.out
gnuplot -persistent res/result.gp
gnuplot -persistent res/result.gp > out-p4-01-arm.svg
*/

#include "p4-arm-helper.h"
#include <ompl/geometric/SimpleSetup.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/geometric/planners/prm/PRM.h>

void ConstructArm(int N)
{
  ArmBase= V3(0.0,0.0,0.0);
  double total_len(0.99*(SizeZ-ArmBase(2)));
  Arm.push_back(TLink(V3(0,0,1), V3(0,0,0.0)));
  for(int i(0); i<N; ++i)
    Arm.push_back(TLink(V3(0,1,0), V3(0,0,total_len/(double)N)));
}

inline bool IsCrossing(const TVector &pa, const TVector &pb, const TVector &po, const double &ro)
{
  using namespace boost::numeric::ublas;
  double lab= norm_2(pb-pa);
  if(lab<=1.0e-6)  return norm_2(po-pa)<ro;
  TVector u= (pb-pa)/lab;
  double t= inner_prod(po-pa, u);
  if(t<0.0)  return norm_2(po-pa)<ro;
  if(t>lab)  return norm_2(po-pb)<ro;
  return norm_2(po-(pa+t*u))<ro;
}

bool IsStateValid(const ob::State *state)
{
  const ob::RealVectorStateSpace::StateType *state_a= state->as<ob::RealVectorStateSpace::StateType>();
  std::vector<double> angles(Arm.size());
  for(size_t i(0); i<Arm.size(); ++i)
    angles[i]= (*state_a)[i];
  std::vector<TVector> jpos;
  ForwardKinematics(Arm, angles, ArmBase, jpos);
  assert(jpos.size()==Arm.size()+1);

  for(size_t i(0); i<jpos.size(); ++i)
  {
    if(jpos[i](0)<0.0 || SizeX<jpos[i](0) ||
      jpos[i](1)<0.0 || SizeY<jpos[i](1) ||
      jpos[i](2)<0.0 || SizeZ<jpos[i](2))  return false;
  }

  for(std::vector<TVector>::const_iterator itr(Obstacles.begin()),last(Obstacles.end()); itr!=last; ++itr)
  {
    TVector po= V3((*itr)(0),(*itr)(1),(*itr)(2));
    for(size_t i(0); i<Arm.size(); ++i)
      if(IsCrossing(jpos[i], jpos[i+1], po, ObstacleRadius))  return false;
  }
  return true;
}

void PlanWithSimpleSetup(void)
{
  int N= Arm.size();

  // construct the state space we are planning in
  ob::StateSpacePtr space(new ob::RealVectorStateSpace(N));

  ob::RealVectorBounds bounds(N);
  bounds.setLow(-M_PI);
  bounds.setHigh(M_PI);
  space->as<ob::RealVectorStateSpace>()->setBounds(bounds);

  og::SimpleSetup ss(space);

  ss.setStateValidityChecker(boost::bind(&IsStateValid, _1));

  ob::ScopedState<ob::RealVectorStateSpace> start(space);
  for(int i(0);i<N;++i)  start[i]= 0.0;
  std::cout << "start: "; start.print(std::cout);

  ob::ScopedState<ob::RealVectorStateSpace> goal(start);
  goal[0]= M_PI*0.25;
  goal[1]= M_PI*0.5;
  std::cout << "goal: "; goal.print(std::cout);

  ss.setStartAndGoalStates(start, goal);

  // specify the planner
  // ob::PlannerPtr planner(new og::PRM(ss.getSpaceInformation()));
  // ss.setPlanner(planner);

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
    PrintArmSolution("res/result.gp", ss.getSolutionPath());
  }
}

int main()
{
  ConstructArm(4);
  CreateMap(50, 0);
  PlanWithSimpleSetup();
  return 0;
}
