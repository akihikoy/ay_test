#ifdef arm_h
#  error DO NOT INCLUDE arm.h TWICE
#endif

#define arm_h

#include "xode.h"

namespace xode
{

void TDynRobot::Create(dWorldID world, dSpaceID space)
{
  Clear();

  dReal cz(0.5);
  dReal len(0.6);

  body_.resize(4);
  link_b_.resize(2);
  link_cy_.resize(2);

  int i(0),ib(0),icy(0);
  {
    dReal side1(len),side2(0.2),side3(0.1);
    body_[i].create (world);
    body_[i].setPosition (-0.5*len,0.0,0.0+cz);
    dMass m;
    m.setBox (1,side1,side2,side3);
    // m.adjust (mass);
    body_[i].setMass (&m);
    link_b_[ib].create (space,side1,side2,side3);
    link_b_[ib].setBody (body_[i]);
    ++i;++ib;
  }
  {
    dReal side1(0.05),side2(0.2),side3(0.2);
    body_[i].create (world);
    body_[i].setPosition (-0.025,0.0,0.15+cz);
    dMass m;
    m.setBox (1,side1,side2,side3);
    // m.adjust (mass);
    body_[i].setMass (&m);
    link_b_[ib].create (space,side1,side2,side3);
    link_b_[ib].setBody (body_[i]);
    ++i;++ib;
  }
  {
    body_[i].create (world);
    body_[i].setPosition (-len,0.0,0.0+cz);
    dMatrix3 R;
    dRFromAxisAndAngle (R,1.0,0.0,0.0,0.5*M_PI);
    body_[i].setRotation (R);
    dMass m;
    // m.setCylinder (1,/*z*/3,/*radius*/0.3,/*length*/0.2);
    dMassSetCylinder (&m,1,/*y*/2,/*radius*/0.05,/*length*/0.2);
    // m.adjust (mass);
    body_[i].setMass (&m);
    link_cy_[icy].create (space,/*radius*/0.05,/*length*/0.2);
    link_cy_[icy].setBody (body_[i]);
    ++i;++icy;
  }

  // ball
  {
    body_[i].create (world);
    body_[i].setPosition (-0.15,0.0,0.15+cz);
    dMatrix3 R;
    dRFromAxisAndAngle (R,1.0,0.0,0.0,0.5*M_PI);
    body_[i].setRotation (R);
    dMass m;
    // m.setCylinder (1,/*z*/3,/*radius*/0.3,/*length*/0.2);
    dMassSetCylinder (&m,1,/*y*/2,/*radius*/0.1,/*length*/0.2);
    // m.adjust (mass);
    body_[i].setMass (&m);
    link_cy_[icy].create (space,/*radius*/0.1,/*length*/0.2);
    link_cy_[icy].setBody (body_[i]);
    ++i;++icy;
  }

  dReal fmax(1000.0);  // NOTE: set zero to control by joint torque

  joint_h_.resize(1);
  joint_f_.resize(2);

  int jh(0),jf(0);

  {
    joint_h_[jh].create (world);
    joint_h_[jh].attach (body_[0],0);
    joint_h_[jh].setAnchor (0.0,0.0,0.0+cz);
    joint_h_[jh].setAxis (0.0,1.0,0.0);
    joint_h_[jh].setParam(dParamFMax,fmax);
    ++jh;
  }
  {
    joint_f_[jf].create (world);
    joint_f_[jf].attach (body_[1],body_[0]);
    joint_f_[jf].set();
    ++jf;
  }
  {
    joint_f_[jf].create (world);
    joint_f_[jf].attach (body_[2],body_[0]);
    joint_f_[jf].set();
    ++jf;
  }
}

}  // end of xode
