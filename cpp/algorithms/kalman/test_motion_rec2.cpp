//-------------------------------------------------------------------------------------------
/*! \file    test_motion_rec.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    Nov.1, 2013

    x++ -slora -oct test_motion_rec.cpp

    rfcomm connect -i 00:09:E7:02:37:2A
    qplot res/obs.dat u 4 res/est_v.dat u 1 w l res/est_p.dat u 1 w l res/est_a.dat u 1 w l
    qplot -3d -s 'set size 1,1.2;set ticslevel 0;set xrange [-1:1];set yrange [-1:1];set zrange[-1:1]' res/box.dat w l -i 0.1
*/
//-------------------------------------------------------------------------------------------
#include "q_ekf2.h"
#include <lora/octave.h>
#include <lora/type_gen_oct.h>
#include <lora/sys.h>
#include <lora/small_classes.h>  // TMovingAverageFilter
#include <boost/bind.hpp>
#include <fstream>
//-------------------------------------------------------------------------------------------
#define NOT_MAIN
#include "ms-motion-recorder.cpp"
//-------------------------------------------------------------------------------------------
namespace loco_rabbits
{
}
//-------------------------------------------------------------------------------------------
using namespace std;
// using namespace boost;
using namespace loco_rabbits;
//-------------------------------------------------------------------------------------------
// #define print(var) PrintContainer((var), #var"= ")
#define print(var) std::cerr<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

const double gravity[]={0.0,0.0,9.8};
// const double gravity[]={0.0,0.0,9.8195};
// const double gravity[]={0.0,0.0,0.0};

TKalmanFilter::TVector noise_mean_q(3,0.0), noise_mean_p(3,0.0);

void SetupEKF(TExtendedKalmanFilter &ekf_q, TExtendedKalmanFilter &ekf_p, TReal dt)
{
  TReal dx(0.00), dz(0.00);
  TKalmanFilter::TVector g(3,0.0);
  g(0)= gravity[0]; g(1)= gravity[1]; g(2)= gravity[2];
  // TKalmanFilter::TMatrix R(GetEye(16)*dx*dx), Q(GetEye(6)*dv*dv);
  TKalmanFilter::TMatrix Rq(GetEye(7)*dx*dx), Qq(GetEye(3)*dz*dz);
  Rq(4,4)= Rq(5,5)= Rq(6,6)= 0.04;
  TKalmanFilter::TMatrix Rp(GetEye(9)*dx*dx), Qp(GetEye(3)*dz*dz);
  Rp(6,6)= Rp(7,7)= Rp(8,8)= 0.01;

  using namespace q_ekf;
  ekf_q.SetStateTransModel(boost::bind(&func_state_trans_q,_1,_2,dt));
  ekf_q.SetObservationModel(boost::bind(&func_observation_q,_1,dt,noise_mean_q));
  ekf_q.SetStateConstraint(boost::bind(&func_constrain_state_q,_1));
  ekf_q.SetG(boost::bind(&func_G_q,_1,_2,dt));
  ekf_q.SetH(boost::bind(&func_H_q,_1,dt));
  ekf_q.SetR(Rq);
  ekf_q.SetQ(Qq);

  ekf_p.SetStateTransModel(boost::bind(&func_state_trans_p,_1,_2,dt));
  ekf_p.SetObservationModel(boost::bind(&func_observation_p,_1,dt,g,noise_mean_p));
  ekf_p.SetStateConstraint(boost::bind(&func_constrain_state_p,_1));
  ekf_p.SetG(boost::bind(&func_G_p,_1,_2,dt));
  ekf_p.SetH(boost::bind(&func_H_p,_1,dt));
  ekf_p.SetR(Rp);
  ekf_p.SetQ(Qp);

  TKalmanFilter::TVector init_mu_q(7,0.0), init_mu_p(9,0.0), init_q(4,0.0);
  init_q(0)= 1.0;
  q_ekf::detail::ASIGN_Q(init_mu_q,init_q);
  // ekf.Initialize(init_mu, GetEye(16));
  ekf_q.Initialize(init_mu_q, GetEye(7)*0.0);
  ekf_p.Initialize(init_mu_p, GetEye(9)*0.0);
}

inline TKalmanFilter::TVector V3(const double &x, const double &y, const double &z)
{
  TKalmanFilter::TVector v(3,0.0);
  v(0)= x; v(1)= y; v(2)= z;
  return v;
}
inline TKalmanFilter::TVector V4(const double &w, const double &x, const double &y, const double &z)
{
  TKalmanFilter::TVector v(4,0.0);
  v(0)= w; v(1)= x; v(2)= y; v(3)= z;
  return v;
}

void DrawBox(std::ostream &os,
    const TKalmanFilter::TVector &p,
    const TKalmanFilter::TVector &q,
    const TKalmanFilter::TVector &v,
    const TKalmanFilter::TVector &w,
    const double &lx, const double &ly, const double &lz)
{
  TKalmanFilter::TMatrix R= q_ekf::detail::QtoR(q);
  os<< (R*V3(lx,ly,lz)+p).transpose() <<std::endl;
  os<< (R*V3(-lx,ly,lz)+p).transpose() <<std::endl;
  os<< (R*V3(-lx,-ly,lz)+p).transpose() <<std::endl;
  os<< (R*V3(lx,-ly,lz)+p).transpose() <<std::endl;
  os<< (R*V3(lx,ly,lz)+p).transpose() <<std::endl;

  os<< (R*V3(lx,ly,-lz)+p).transpose() <<std::endl;
  os<< (R*V3(-lx,ly,-lz)+p).transpose() <<std::endl;
  os<< (R*V3(-lx,-ly,-lz)+p).transpose() <<std::endl;
  os<< (R*V3(lx,-ly,-lz)+p).transpose() <<std::endl;
  os<< (R*V3(lx,ly,-lz)+p).transpose() <<std::endl;

  os<< std::endl;
  os<< (p).transpose() <<std::endl;
  os<< (R*v+p).transpose() <<std::endl;

  os<< std::endl;
  os<< (p).transpose() <<std::endl;
  os<< (R*w+p).transpose() <<std::endl;
}

int main(int argc, char**argv)
{
  ms_motion_rec::TMotionRec motion_sensor;

  motion_sensor.Connect("/dev/rfcomm0");
  if(motion_sensor.IsConnected())
  {
    LMESSAGE("Connected");
  }
  else
  {
    LERROR("Failed to connect");
    return -1;
  }
  motion_sensor.Reset();

  print(motion_sensor.GetID());
  #if 1
  if(motion_sensor.GetSensorType())
    print(serial::PrintBuffer(motion_sensor.SensorType(),8));

  print((int)motion_sensor.GetGravityOffsetKind());
  if(motion_sensor.GetCalibrationValue())
    print(serial::PrintVector(motion_sensor.CalibrationValue(),motion_sensor.CalibrationValue()+8));

  /*! Set sensor type (internal accelerometer type and internal gyro type);
      use ms_motion_rec::ACC2G or ACC6G for acc_type,
      ms_motion_rec::GYRO500 or GYRO2000 ro gyro_type  */
  motion_sensor.SetSensorType(ms_motion_rec::ACC2G, ms_motion_rec::GYRO500);

  if(motion_sensor.GetSensorType())
    print(serial::PrintBuffer(motion_sensor.SensorType(),8));

  LMESSAGE("Calibrating..");
  motion_sensor.Calibrate(ms_motion_rec::goGravityZ);
  // motion_sensor.Calibrate(ms_motion_rec::goNone);
  LMESSAGE("  done.");

  print((int)motion_sensor.GetGravityOffsetKind());
  if(motion_sensor.GetCalibrationValue())
    print(serial::PrintVector(motion_sensor.CalibrationValue(),motion_sensor.CalibrationValue()+8));
  #endif

  //! Set sampling cycle (1-100 msec)
  // int cycle(100);
  int cycle(5), ekf_cycle(4);
  motion_sensor.SetSamplingCycle(cycle);
  motion_sensor.StartSampling();

  double t_start(GetCurrentTime());
  int n_total(0);

  // filter for observation
  bool init_flag(true);
  TMovingAverageFilter<TKalmanFilter::TVector> sensor_ang_vel_maf, sensor_acc_maf;

  // setup kalman filter
  TExtendedKalmanFilter ekf_q, ekf_p;
  double ekf_dt= (double)(cycle*ekf_cycle)*1.0e-3;
  SetupEKF(ekf_q, ekf_p, ekf_dt);
  TKalmanFilter::TVector est_position, est_v, est_acc;
  TKalmanFilter::TVector est_quaternion;
  TKalmanFilter::TVector u, sensor_ang_vel(3,0.0), sensor_acc(3,0.0);

  ofstream ofs_obs("res/obs.dat");
  ofstream ofs_est_p("res/est_p.dat");
  ofstream ofs_est_v("res/est_v.dat");
  ofstream ofs_est_q("res/est_q.dat");
  ofstream ofs_est_w("res/est_w.dat");
  ofstream ofs_est_a("res/est_a.dat");
  ofstream ofs_misc("res/misc.dat");
  // for(int i(0);i<100;++i)
  int idx(0), skip_idx(0);

// noise calibration:
{LMESSAGE("Noise calibration..");
noise_mean_q= TKalmanFilter::TVector(3,0.0);
noise_mean_p= TKalmanFilter::TVector(3,0.0);
double c(0);
for(int i(0); i<50; ++i)
{
  size_t NB(40);
  const std::vector<double> &z_set= motion_sensor.GetSamples(NB);
  if(z_set.size()/8==NB)
  {
    for(std::vector<double>::const_iterator z(z_set.begin()),z_last(z_set.end()); z!=z_last; z+=8)
    {
      // cout<<z[0]<<" "<<z[1]<<" "<<z[2]<<"  "<<z[4]<<" "<<z[5]<<" "<<z[6]<<endl;
      for(int r(0);r<3;++r)  sensor_ang_vel(r)= z[r+4];
      for(int r(0);r<3;++r)  sensor_acc(r)= z[r];
      noise_mean_q+= sensor_ang_vel;
      noise_mean_p+= sensor_acc;
      ++c;
    }
  }
  usleep(100*1000);
}
noise_mean_q= noise_mean_q/(double)c;
noise_mean_p= noise_mean_p/(double)c;
noise_mean_p(2)-= gravity[2];
ofs_misc<<noise_mean_q.transpose()<<" "<<noise_mean_p.transpose()<<endl;
SetupEKF(ekf_q, ekf_p, ekf_dt);
LMESSAGE("  done.");}

  while(true)
  {
    size_t NB(40);
    // const double *z= motion_sensor.GetSample();
    const std::vector<double> &z_set= motion_sensor.GetSamples(NB);
// ++idx;if(idx%1000==0)  motion_sensor.Clear();
    if(z_set.size()/8==NB)
    {
      for(std::vector<double>::const_iterator z(z_set.begin()),z_last(z_set.end()); z!=z_last; z+=8)
      {
        // cout<<z[0]<<" "<<z[1]<<" "<<z[2]<<"  "<<z[4]<<" "<<z[5]<<" "<<z[6]<<endl;
        for(int r(0);r<3;++r)  sensor_ang_vel(r)= z[r+4];
        for(int r(0);r<3;++r)  sensor_acc(r)= z[r];

// correration of the data
// for(int r(0);r<3;++r)
// {
  // if(real_fabs(v_obs(r))<2.1*motion_sensor.SensorCoefficient(r+4))
    // v_obs(r)= 0.0;
// }
// for(int r(0);r<3;++r)
// {
  // if(real_fabs(v_obs(r+3))<2.1*motion_sensor.SensorCoefficient(r))
    // v_obs(r+3)= 0.0;
// }

// set ang-vel zero:
// v_obs(0)=v_obs(1)=v_obs(2)=0.0;
// set acc zero:
// v_obs(3)=v_obs(4)=v_obs(5)=0.0;

        const int N_MAF(4);
        if(init_flag)
        {
          sensor_ang_vel_maf.Initialize(N_MAF,TKalmanFilter::TVector(3,0.0));
          sensor_acc_maf.Initialize(N_MAF,TKalmanFilter::TVector(3,0.0));
          skip_idx= N_MAF;
          init_flag=false;
        }
        sensor_ang_vel_maf.Step(sensor_ang_vel);
        sensor_acc_maf.Step(sensor_acc);
        if(skip_idx>0)  {--skip_idx; continue;}

        ++idx;
        if(idx%ekf_cycle!=0)  continue;

        sensor_ang_vel= sensor_ang_vel_maf();
        sensor_acc= sensor_acc_maf();

        // EKF update and estimation
        ekf_q.Update(u, sensor_ang_vel);
        est_quaternion= q_ekf::detail::EXT_Q(ekf_q.GetMu());

        ekf_p.Update(u, q_ekf::detail::QtoR(est_quaternion)*sensor_acc);

        est_position= q_ekf::detail::EXT_P(ekf_p.GetMu());
        est_v= q_ekf::detail::EXT_V(ekf_p.GetMu());
        est_acc= q_ekf::detail::EXT_A(ekf_p.GetMu());

        print(sensor_ang_vel.transpose());
        print(sensor_acc.transpose());
        print(est_acc.transpose());
        print(est_v.transpose());
        print(est_position.transpose());
        print(est_quaternion.transpose());

        ofs_obs<<sensor_ang_vel.transpose()<<" "<<sensor_acc.transpose()<<endl;
        ofs_est_p<<est_position.transpose()<<endl;
        ofs_est_v<<est_v.transpose()<<endl;
        ofs_est_q<<est_quaternion.transpose()<<endl;
        ofs_est_w<<q_ekf::detail::EXT_W(ekf_q.GetMu()).transpose()<<endl;
        ofs_est_a<<q_ekf::detail::EXT_A(ekf_p.GetMu()).transpose()<<endl;
      }
      n_total+= z_set.size()/8;
    }  // z_set.size()==NB
    else
    {
      LERROR("failed to sample");
    }

    cerr<<GetCurrentTime()-t_start<<" "<<n_total<<endl;
    cerr<<"FPS: "<<(double)n_total/(GetCurrentTime()-t_start)<<endl;

{ofstream ofs_box("res/box.dat");
DrawBox(ofs_box,
  // q_ekf::detail::EXT_P(ekf_p.GetMu()),
  V3(0,0,0),
  q_ekf::detail::EXT_Q(ekf_q.GetMu()),
  // V4(1,0,0,0),
  q_ekf::detail::EXT_V(ekf_p.GetMu()),
  // q_ekf::detail::EXT_A(ekf_p.GetMu()),
  // V3(0,0,0),
  // q_ekf::detail::EXT_ZW(sensor_ang_vel),
  q_ekf::detail::EXT_W(ekf_q.GetMu()),
  // q_ekf::detail::EXT_ZA(sensor_acc)*0.1,
  0.1,0.08,0.05);
ofs_box.close();}

    int c=KBHit();
    if(c=='x')  break;
    if(c=='r')  {SetupEKF(ekf_q, ekf_p, ekf_dt); init_flag= true;}
  }
  cerr<<GetCurrentTime()-t_start<<" "<<n_total<<endl;
  cerr<<"FPS: "<<(double)n_total/(GetCurrentTime()-t_start)<<endl;

  motion_sensor.StopSampling();
  motion_sensor.Disconnect();
  return 0;
}
//-------------------------------------------------------------------------------------------
