#ifndef CAL_DIST_H
#define CAL_DIST_H

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>

// #include <irrlicht.h>
//dev
// #include "tclap/CmdLine.h"
#include <iostream>
#include <iomanip>
//#include "QuinticsSolver.hpp"
#include <cmath>
#include <cstdio>
#include <vector>


// using namespace irr;
// using namespace core;
// using namespace scene;
// using namespace video;
// using namespace io;
// using namespace gui;

// FIXME: replace float by double



// #define isnan(x) _isnan(x)
// #define isinf(x) (!_finite(x))
#define fpu_error(x) (isinf(x) || isnan(x))

// using namespace std;  // FIXME(BAD CODE)
// using namespace boost::numeric::ublas;  // FIXME(BAD CODE)


// "Eager" evaluation cross product
template <class V1, class V2>
boost::numeric::ublas::vector<typename boost::numeric::ublas::promote_traits<typename V1::value_type,
                                                                             typename V2::value_type>::promote_type>
cross_prod(const V1& lhs, const V2& rhs)
{
  BOOST_UBLAS_CHECK(lhs.size() == 3, boost::numeric::ublas::external_logic());
  BOOST_UBLAS_CHECK(rhs.size() == 3, boost::numeric::ublas::external_logic());

  typedef typename boost::numeric::ublas::promote_traits<typename V1::value_type,
                                                         typename V2::value_type>::promote_type promote_type;

  boost::numeric::ublas::vector<promote_type> temporary(3);

  temporary(0) = lhs(1) * rhs(2) - lhs(2) * rhs(1);
  temporary(1) = lhs(2) * rhs(0) - lhs(0) * rhs(2);
  temporary(2) = lhs(0) * rhs(1) - lhs(1) * rhs(0);

  return temporary;
}



template<typename T>
inline T Sq(const T &x)  {return x*x;}

struct vector3df
{
  typedef boost::numeric::ublas::vector<double> v_type;
  v_type v;  // FIXME: should be private
  vector3df() : v(3,0.0) {}
  vector3df(const v_type &_v) : v(_v) {}
  vector3df(const double &x,const double &y,const double &z) : v(3) {v(0)=x;v(1)=y;v(2)=z;}

  void set(const double &x,const double &y,const double &z)  {v.resize(3); v(0)=x;v(1)=y;v(2)=z;}

  double& X()  {return v(0);}
  double& Y()  {return v(1);}
  double& Z()  {return v(2);}
  const double& X() const {return v(0);}
  const double& Y() const {return v(1);}
  const double& Z() const {return v(2);}

  double getLength() const {return norm_2(v);}
  double getLengthSQ() const {return Sq(norm_2(v));}  // FIXME

  double getDistanceFromSQ(const vector3df& other) const
  {
      return vector3df(X() - other.X(), Y() - other.Y(), Z() - other.Z()).getLengthSQ();
  }

  double dotProduct(const vector3df &a) const {return inner_prod(v,a.v);}
  vector3df crossProduct(const vector3df &a) const {return vector3df(cross_prod(v,a.v));}

  inline bool isBetweenPoints(const vector3df& begin, const vector3df& end) const;

// FIXME: do not use degrees but use radian
  void rotateXZBy(double degrees, const vector3df& center=vector3df())
  {
    typedef double f64;
    degrees *= M_PI/180.0;
    f64 cs = cos(degrees);
    f64 sn = sin(degrees);
    X() -= center.X();
    Z() -= center.Z();
    set((X()*cs - Z()*sn), Y(), (X()*sn + Z()*cs));
    X() += center.X();
    Z() += center.Z();
  }
  void rotateXYBy(double degrees, const vector3df& center=vector3df())
  {
    typedef double f64;
    degrees *= M_PI/180.0;
    f64 cs = cos(degrees);
    f64 sn = sin(degrees);
    X() -= center.X();
    Y() -= center.Y();
    set((X()*cs - Y()*sn), (X()*sn + Y()*cs), Z());
    X() += center.X();
    Y() += center.Y();
  }
  void rotateYZBy(double degrees, const vector3df& center=vector3df())
  {
    typedef double f64;
    degrees *= M_PI/180.0;
    f64 cs = cos(degrees);
    f64 sn = sin(degrees);
    Z() -= center.Z();
    Y() -= center.Y();
    set(X(), (Y()*cs - Z()*sn), (Y()*sn + Z()*cs));
    Z() += center.Z();
    Y() += center.Y();
  }

  operator v_type&() {return v;}
  operator const v_type&() const {return v;}
};

inline vector3df operator+(const vector3df&a, const vector3df&b)
{
  return vector3df(a.v+b.v);
}
inline vector3df operator-(const vector3df&a, const vector3df&b)
{
  return vector3df(a.v-b.v);
}
inline vector3df operator*(const double&a, const vector3df&b)
{
  return vector3df(a*b.v);
}
inline vector3df operator*(const vector3df&a, const double&b)
{
  return vector3df(a.v*b);
}
inline vector3df operator/(const vector3df&a, const double&b)
{
  return vector3df(a.v/b);
}
inline bool operator==(const vector3df&a, const vector3df&b)
{
  return a.v==b.v;
}
inline bool operator!=(const vector3df&a, const vector3df&b)
{
  return a.v!=b.v;
}



inline bool vector3df::isBetweenPoints(const vector3df& begin, const vector3df& end) const
{
  const double f = (end - begin).getLengthSQ();
  return getDistanceFromSQ(begin) <= f &&
      getDistanceFromSQ(end) <= f;
}



struct line3df
{
  vector3df start;
  vector3df end;
  line3df() {}
  line3df(const vector3df &s, const vector3df &e) : start(s), end(e) {}
  const double getLength() const {return (start-end).getLength();}
  const double getLengthSQ() const {return (start-end).getLengthSQ();}
  bool isPointBetweenStartAndEnd(const vector3df &p) const {return p.isBetweenPoints(start, end);}
};

class CalDistance
{
public:

  inline vector3df getNormalizedVec(const vector3df&);
  std::vector<vector3df> GetClosestPointAndDist(const line3df &, const vector3df &, float &);
  std::vector<vector3df> GetClosestPointAndDist(const line3df&, const line3df&, float &);
  std::vector<vector3df> GetClosestPointAndDist(const vector3df& ,//point
                      const vector3df&,//center
                      const float &, //radius
                      const float&, const float& ,
                      const float& ,float&); //to debug
  std::vector<vector3df> GetClosestPointAndDist(const line3df& , //reference line
                      const vector3df& , //center
                      const float &, //radius
                      const float& , const float& ,
                      const float& ,float&  ); //to debug
  std::vector<vector3df> GetClosestPointAndDist(const vector3df& , //reference circle
                      const vector3df& , //center
                      const float& , //radius ref circle
                      const float& , //radius
                      const float& , const float& ,
                      const float& , const float& ,
                      const float& , const float& ,
                      float&  ); //to debug
  std::vector<vector3df> GetClosestPointAndDist(const vector3df& , //begin cylinder
                      const vector3df& , //
                      const vector3df& , //begin cylinder
                      const vector3df& , //begin cylinder
                      const float& , //radius ref circle
                      const float& , //radius
                      const float& , const float& ,
                      const float& , const float& ,
                      const float& , const float& ,
                      float& );
  ///sphere to cylinder
  std::vector<vector3df> GetClosestPointAndDist(const vector3df& C_1, //begin cylinder
                    const vector3df& C_2, //
            const vector3df& center, //sphere center
            const float& Rcyl, //radius cylinder
            const float& Rsph, //radius sphere
            const float& rotX1,
            const float& rotY1,
            const float& rotZ1,
            float& dist) ;

  /// sphere to cylinder extremes (like sphere to sphere)
  std::vector<vector3df> GetClosestPointAndDist(const vector3df& C_1, //begin cylinder
                                 const vector3df& C_2, //
                                 const vector3df& center, //sphere center
                                 const float& cyl_R, //radius cylinder
                                 const float& sph_R, //radius sphere
                                 float& dist);
  /// sphere to cylinder extremes (like sphere to sphere), for debugging //not efficient due to double calculation
  std::vector<vector3df> GetClosestPointAndDist(const vector3df& human_cylinder_C1, //begin cylinder
                            const vector3df& human_cylinder_C2, //
                            const vector3df& robot_cylinder_C1, //begin cylinder
                            const vector3df& robot_cylinder_C2, //
                            const float& human_sphere_radius, //radius of sphere
                            const float& robot_sphere_radius,
                            float& dist);

};

#endif