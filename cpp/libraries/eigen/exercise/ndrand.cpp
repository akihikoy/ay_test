#include <Eigen/Dense>
#include <cstdlib>

template<typename T>
inline T URand()
{
  return static_cast<T>(std::rand())/static_cast<T>(RAND_MAX);
}
template<typename T>
inline T ApproxNRand()
{
  T r(0.0l);
  for(int i(0);i<12;++i) r+=URand<T>();
  return r-6.0l;
}
template<typename T>
inline T DummyApproxNRand(const T &)
{
  return ApproxNRand<T>();
}
class TMultiNRand
{
public:
  TMultiNRand(const Eigen::VectorXd &mean, const Eigen::MatrixXd &Cov)  {Init(mean,Cov);}
  void Init(const Eigen::VectorXd &mean, const Eigen::MatrixXd &Cov)
    {
      mean_= mean;
      L_= Cov.llt().matrixL();
    }
  Eigen::VectorXd operator()()
    {
      Eigen::VectorXd x(mean_.size());
      x= x.unaryExpr(&DummyApproxNRand<double>);
      return L_*x+mean_;
    }
private:
  Eigen::VectorXd mean_;
  Eigen::MatrixXd L_;
};

#include <iostream>

using namespace std;
using namespace Eigen;

int main()
{
  Vector2d mean(-3.0, 1.0);
  Matrix2d Cov;
  Cov<< 4.0, -1.0,
        -1.0, 2.5;
  TMultiNRand mnrand(mean,Cov);

  Vector2d sample(0.0,0.0), sum(0.0,0.0);
  Matrix2d sum2(Matrix2d::Zero());
  double N(10000);
  for(int i(0);i<N;++i)
  {
    sample= mnrand();
    sum+= sample;
    sum2+= sample * sample.transpose();
    cout<<sample.transpose()<<endl;
  }
  cerr<<"mean= "<<(sum/N).transpose()<<endl;
  cerr<<"Cov= \n"<<sum2/N - mean*mean.transpose()<<endl;
  return 0;
}
