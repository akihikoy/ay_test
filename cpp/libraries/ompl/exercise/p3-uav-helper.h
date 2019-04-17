// Utility for the UAV planning problem

#include <ompl/geometric/SimpleSetup.h>
#include <ompl/base/spaces/SE3StateSpace.h>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <cassert>
#include <iostream>
#include <cmath>
#include <fstream>
#include <vector>

namespace ob = ompl::base;
namespace og = ompl::geometric;

template<typename T> inline T Sq(const T &x)  {return x*x;}

typedef boost::numeric::ublas::matrix<double> TMatrix;
typedef boost::numeric::ublas::vector<double> TVector;

static const double SizeX(6),SizeY(5),SizeZ(5);

std::vector<TVector>  Obstacles;
static const double ObstacleRadius(0.5);

static const double RobotX(0.5),RobotY(0.4),RobotZ(0.3);
static const double RobotRadius(std::sqrt(0.25*Sq(RobotX)+0.25*Sq(RobotY)+0.25*Sq(RobotZ)));

// Generate a vector with 3-dim
inline TVector V3(const double &x, const double &y, const double &z)
{
  TVector v(3);  v(0)= x; v(1)= y; v(2)= z;
  return v;
}
// Convert a quaternion to a rotation matrix
inline TMatrix QtoR(const double &qx, const double &qy, const double &qz, const double &qw)
{
  TMatrix M(3,3);
  M(0,0)= qw*qw+qx*qx-qy*qy-qz*qz; M(0,1)= 2.0*(qx*qy-qw*qz);       M(0,2)= 2.0*(qx*qz+qw*qy);
  M(1,0)= 2.0*(qx*qy+qw*qz);       M(1,1)= qw*qw-qx*qx+qy*qy-qz*qz; M(1,2)= 2.0*(qy*qz-qw*qx);
  M(2,0)= 2.0*(qx*qz-qw*qy);       M(2,1)= 2.0*(qy*qz+qw*qx);       M(2,2)= qw*qw-qx*qx-qy*qy+qz*qz;
  return M;
}
// Convert a OMPL's quaternion to a rotation matrix
inline TMatrix QtoR(const ob::SO3StateSpace::StateType &rot)
{
  return QtoR(rot.x, rot.y, rot.z, rot.w);
}

// Standard quaternion multiplication: q= q0 * q1
inline void QuaternionProduct(ob::SO3StateSpace::StateType &q, const ob::SO3StateSpace::StateType &q0, const ob::SO3StateSpace::StateType &q1)
{
  q.x = q0.w*q1.x + q0.x*q1.w + q0.y*q1.z - q0.z*q1.y;
  q.y = q0.w*q1.y + q0.y*q1.w + q0.z*q1.x - q0.x*q1.z;
  q.z = q0.w*q1.z + q0.z*q1.w + q0.x*q1.y - q0.y*q1.x;
  q.w = q0.w*q1.w - q0.x*q1.x - q0.y*q1.y - q0.z*q1.z;
}

std::ostream& operator<<(std::ostream &os, const TVector &x)
{
  for(std::size_t i(0); i<x.size(); ++i)
    os<<" "<<x(i);
  return os;
}

/* Save a sequence of box on ``path'' into file that is gnuplot-compatible.
    The path should be a sequence of SE(3) state. The box size is ``(sizex,sizey,sizez)''.
    The parameter ``skip'' is an interval to sample from ``path'' (1 for every sample). */
void PrintBoxSequence(const char *filename, const og::PathGeometric &path, const double &sizex, const double &sizey, const double &sizez, int skip=1)
{
  using namespace std;
  using namespace boost::numeric::ublas;
  ofstream ofs(filename);
  for(size_t i(0); i<path.getStateCount(); i+=skip)
  {
    const ob::SE3StateSpace::StateType *s= path.getState(i)->as<ob::SE3StateSpace::StateType>();
    TVector pos(3), d(3);
    TMatrix R= QtoR(s->rotation());
    pos(0)= s->getX(); pos(1)= s->getY(); pos(2)= s->getZ();
    ofs<<pos+prod(R,V3( sizex, sizey, sizez))<<endl;
    ofs<<pos+prod(R,V3( sizex,-sizey, sizez))<<endl;
    ofs<<pos+prod(R,V3(-sizex,-sizey, sizez))<<endl;
    ofs<<pos+prod(R,V3(-sizex, sizey, sizez))<<endl;
    ofs<<pos+prod(R,V3( sizex, sizey, sizez))<<endl;
    ofs<<pos+prod(R,V3( sizex, sizey,-sizez))<<endl;
    ofs<<pos+prod(R,V3( sizex,-sizey,-sizez))<<endl;
      ofs<<pos+prod(R,V3( sizex,-sizey,sizez))<<endl;
      ofs<<pos+prod(R,V3( sizex,-sizey,-sizez))<<endl;
    ofs<<pos+prod(R,V3(-sizex,-sizey,-sizez))<<endl;
      ofs<<pos+prod(R,V3(-sizex,-sizey,sizez))<<endl;
      ofs<<pos+prod(R,V3(-sizex,-sizey,-sizez))<<endl;
    ofs<<pos+prod(R,V3(-sizex, sizey,-sizez))<<endl;
      ofs<<pos+prod(R,V3(-sizex, sizey,sizez))<<endl;
      ofs<<pos+prod(R,V3(-sizex, sizey,-sizez))<<endl;
    ofs<<pos+prod(R,V3( sizex, sizey,-sizez))<<endl;
      ofs<<pos+prod(R,V3( sizex, sizey,sizez))<<endl;
    ofs<<endl<<endl;
  }
}

/* Generate ``num'' shperes and store them into ``Obstacles''
    where each center is decided randomly. */
void CreateMap(int num, long seed)
{
  using namespace std;
  srand(seed);
  Obstacles.resize(num);
  for(int i(0); i<num; ++i)
  {
    Obstacles[i].resize(3);
    Obstacles[i](0)= SizeX*(double)rand()/(double)RAND_MAX;
    Obstacles[i](1)= SizeY*(double)rand()/(double)RAND_MAX;
    Obstacles[i](2)= SizeZ*(double)rand()/(double)RAND_MAX;
  }
}

/* Print every center shperes into a file "res/map.dat". */
void PrintMap()
{
  using namespace std;
  ofstream ofs("res/map.dat");
  for(vector<TVector>::const_iterator itr(Obstacles.begin()),last(Obstacles.end()); itr!=last; ++itr)
    ofs<<(*itr)<<endl;
}

/* Print the planning result into a file.
    The resulting file is a gnuplot script that plots the path,
    the sequence of box on the path, and the obstacles.
    ``path'': stored into "res/path.dat",
    sequence of box on ``path'': stored into "res/frame_all.dat",
    obstacles: stored into the resulting script.
    Usage:  gnuplot -persistent filename */
void PrintSolution(const char *filename, const og::PathGeometric &path, int skip=1)
{
  using namespace std;
  ofstream ofs(filename);
  {
    ofstream ofs("res/path.dat");
    path.printAsMatrix(ofs);
  }
  PrintBoxSequence("res/frame_all.dat", path, RobotX, RobotY, RobotZ, skip);
  ofs<<"\
#set terminal png size 800, 640 transparent                     \n\
#set terminal svg size 1200 780 fname 'Trebuchet MS' fsize 24   \n\
set xlabel 'x'         \n\
set ylabel 'y'         \n\
set zlabel 'z'         \n\
set hidden3d           \n\
set ticslevel 0        \n\
set size 0.7,1         \n\
set parametric         \n\
set urange [0:6.28]    \n\
set vrange [0:6.28]    \n\
set isosample 8,8      \n\
set samples 10         \n\
r= "<<ObstacleRadius<<endl;
  ofs<<"splot \\"<<endl;
  for(vector<TVector>::const_iterator itr(Obstacles.begin()),last(Obstacles.end()); itr!=last; ++itr)
  {
    const double &ox((*itr)(0)), &oy((*itr)(1)), &oz((*itr)(2));
    ofs<<"  "<<"r*cos(u)*cos(v)+"<<ox
        <<",r*sin(u)*cos(v)+"<<oy
        <<",r*sin(v)+"<<oz<<" w l lt 1 lw 0.2 t '',"<<"\\"<<endl;
  }
  ofs<<"'res/frame_all.dat' w l lt 3, \\"<<endl;
  ofs<<"'res/path.dat' w l lt 4"<<endl;
}
