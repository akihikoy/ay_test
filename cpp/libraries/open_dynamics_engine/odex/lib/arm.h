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

  body_.resize(5);
  link_b_.resize(1);
  link_ca_.resize(2);
  link_cy_.resize(2);

  int i(0),ib(0),ica(0),icy(0);

  // base1
  {
    body_[i].create (world);
    body_[i].setPosition (0.0,0.0,0.05);
    dMass m;
    // m.setCylinder (1,/*z*/3,/*radius*/0.3,/*length*/0.2);
    dMassSetCylinder (&m,1,/*z*/3,/*radius*/0.4,/*length*/0.1);
    // m.adjust (mass);
    body_[i].setMass (&m);
    link_cy_[icy].create (space,/*radius*/0.4,/*length*/0.1);
    link_cy_[icy].setBody (body_[i]);
    ++i;++icy;
  }
  // base2
  {
    body_[i].create (world);
    body_[i].setPosition (0.0,0.0,0.15);
    dMass m;
    // m.setCylinder (1,/*z*/3,/*radius*/0.3,/*length*/0.2);
    dMassSetCylinder (&m,1,/*z*/3,/*radius*/0.3,/*length*/0.1);
    // m.adjust (mass);
    body_[i].setMass (&m);
    link_cy_[icy].create (space,/*radius*/0.3,/*length*/0.1);
    link_cy_[icy].setBody (body_[i]);
    ++i;++icy;
  }
  // arm1
  {
    body_[i].create (world);
    body_[i].setPosition (0.0,0.0,0.5);
    dMass m;
    m.setCapsule (1,/*z*/3,/*radius*/0.1,/*length*/0.6-0.1*2.0);
    // m.adjust (mass);
    body_[i].setMass (&m);
    link_ca_[ica].create (space,/*radius*/0.1,/*length*/0.6-0.1*2.0);
    link_ca_[ica].setBody (body_[i]);
    ++i;++ica;
  }
  // arm2
  {
    body_[i].create (world);
    body_[i].setPosition (0.0,0.0,1.1);
    dMass m;
    m.setCapsule (1,/*z*/3,/*radius*/0.1,/*length*/0.6-0.1*2.0);
    // m.adjust (mass);
    body_[i].setMass (&m);
    link_ca_[ica].create (space,/*radius*/0.1,/*length*/0.6-0.1*2.0);
    link_ca_[ica].setBody (body_[i]);
    ++i;++ica;
  }
  // hand (fixed to arm2)
  {
    dReal side(0.25),side2(0.2);
    body_[i].create (world);
    body_[i].setPosition (0.0,0.0,1.4);
    dMass m;
    m.setBox (1,side,side,side2);
    // m.adjust (mass);
    body_[i].setMass (&m);
    link_b_[ib].create (space,side,side,side2);
    link_b_[ib].setBody (body_[i]);
    ++i;++ib;
  }

  dReal fmax(1000.0);  // NOTE: set zero to control by joint torque

  joint_h_.resize(3);
  joint_f_.resize(2);

  int jh(0),jf(0);

  // base1
  {
    joint_f_[jf].create (world);
    joint_f_[jf].attach (0,body_[0]);
    joint_f_[jf].set();
    ++jf;
  }
  // base2
  {
    joint_h_[jh].create (world);
    joint_h_[jh].attach (body_[1],body_[0]);
    joint_h_[jh].setAnchor (0.0,0.0,0.1);
    joint_h_[jh].setAxis (0.0,0.0,1.0);
    joint_h_[jh].setParam(dParamFMax,fmax);
    ++jh;
  }
  // arm1
  {
    joint_h_[jh].create (world);
    joint_h_[jh].attach (body_[2],body_[1]);
    joint_h_[jh].setAnchor (0.0,0.0,0.2);
    joint_h_[jh].setAxis (0.0,1.0,0.0);
    joint_h_[jh].setParam(dParamFMax,fmax);
    ++jh;
  }
  // arm2
  {
    joint_h_[jh].create (world);
    joint_h_[jh].attach (body_[3],body_[2]);
    joint_h_[jh].setAnchor (0.0,0.0,0.8);
    joint_h_[jh].setAxis (0.0,1.0,0.0);
    joint_h_[jh].setParam(dParamFMax,fmax);
    ++jh;
  }
  // hand (fixed to arm2)
  {
    joint_f_[jf].create (world);
    joint_f_[jf].attach (body_[4],body_[3]);
    joint_f_[jf].set();
    ++jf;
  }
}

const dReal* TDynRobot::GetHandPos()
{
  return link_b_.back().getPosition();
}
const dReal* TDynRobot::GetHandRot()
{
  return link_b_.back().getRotation();
}


}  // end of xode
