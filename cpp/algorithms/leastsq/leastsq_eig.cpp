//-------------------------------------------------------------------------------------------
/*! \file    leastsq_eig.cpp
    \brief   Least squares with Eigen
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    Nov.25, 2013

    Usage:
      x++ -eig -slora leastsq.cpp
      ./a.out
      ./a.out ls 100 5 5 false
      ./a.out svd 100 10 10 false
      ./a.out svd 100 25 40 false
      qplot -3d res/sample.dat pt 7 ps 2 res/target.dat w l res/test.dat w l
*/
//-------------------------------------------------------------------------------------------
#include <lora/rand.h>
#include <lora/sys.h>
#include <lora/math.h>
#include <lora/type_gen.h>
#include <fstream>
#include <sstream>
#include <Eigen/Dense>
//-------------------------------------------------------------------------------------------
namespace loco_rabbits
{

typedef Eigen::VectorXd TVector;
typedef Eigen::MatrixXd TMatrix;

int NX(5),NY(5);
// const int NX(10),NY(10);
// const int NX(25),NY(40);

TReal Gaussian(const TVector &x, const TVector &mu, const TMatrix &Sigma)
{
  TReal ex= real_exp(-0.5l*(x-mu).transpose()*Sigma.inverse()*(x-mu));
  // return ex/real_sqrt(real_pow(2.0*REAL_PI,GenSize(mu))*Sigma.determinant().value());
  return ex;
}

TVector GetFeature(const TVector &x)
{
  TVector f(NX*NY);
  TVector mu(TVector::Zero(2));
  TMatrix Sigma(TMatrix::Zero(2,2));
  Sigma(0,0)=2.0/TReal(NX*NX); Sigma(1,1)=2.0/TReal(NY*NY);
  for(int ix(0); ix<NX; ++ix)
  {
    mu(0)= -1.0 + 2.0/(TReal(NX)-1.0)*TReal(ix);
    for(int iy(0); iy<NY; ++iy)
    {
      mu(1)= -1.0 + 2.0/(TReal(NY)-1.0)*TReal(iy);
      f(ix*NY+iy)= Gaussian(x, mu, Sigma);
    }
  }
  return f;
}

}
//-------------------------------------------------------------------------------------------
using namespace std;
// using namespace boost;
using namespace loco_rabbits;
//-------------------------------------------------------------------------------------------
// #define print(var) PrintContainer((var), #var"= ")
// #define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

TReal TargetFunc(const TVector &x)
{
  // return x(0)*real_sin(3.0*x(1));
  // return 3.0-(x(0)*x(0)+x(1)*x(1));

  // func1:
  // return 3.0-(x(0)*x(0)+Square(real_sin(3.0*x(1))));
  // func2:
  if(x(0)*x(0)+x(1)*x(1) < 0.25) return 1.0; else return 0.0;
}

enum TSolver {sLeastSquares=0, sSVD};
int main(int argc, char**argv)
{
  TSolver solver= sLeastSquares;
  // TSolver solver= sSVD;
  int N_sample(90);
  bool report(false);

  if(argc>1)
  {
    string s(argv[1]);
    if(s=="ls")  solver= sLeastSquares;
    else if(s=="svd")  solver= sSVD;
    else {cerr<<"error."<<endl; return -1;}
  }
  if(argc>2)
  {
    stringstream ss(argv[2]);
    ss >> N_sample;
  }
  if(argc>3)
  {
    stringstream ss(argv[3]);
    ss >> NX;
  }
  if(argc>4)
  {
    stringstream ss(argv[4]);
    ss >> NY;
  }
  if(argc>5)
  {
    string s(argv[5]);
    if(s=="true")  report= true;
    else if(s=="false")  report= false;
    else {cerr<<"error."<<endl; return -1;}
  }

  cerr<<"training data.."<<endl;
  Srand();
  TMatrix Theta(TMatrix::Zero(N_sample, NX*NY));
  TVector V(TVector::Zero(N_sample));
  {
    ofstream ofs_sample("res/sample.dat");
    TVector x(TVector::Zero(2)),f;
    for(int i(0); i<N_sample; ++i)
    {
      x(0)= Rand(-1.0,1.0);
      x(1)= Rand(-1.0,1.0);
      f= GetFeature(x);
      for(int r(0); r<GenSize(f); ++r)  Theta(i,r)=f(r);
      V(i)= TargetFunc(x) + 0.1*Rand(-1.0,1.0);
      ofs_sample<<x.transpose()<<" "<<V(i)<<endl;
    }
  }

  cerr<<"target func.."<<endl;
  {
    ofstream ofs_target("res/target.dat");
    TVector x(TVector::Zero(2));
    for(x(0)=-1.3; x(0)<=1.3; x(0)+=0.05)
    {
      for(x(1)=-1.3; x(1)<=1.3; x(1)+=0.05)
      {
        ofs_target<<x.transpose()<<"  "<<TargetFunc(x)<<endl;
      }
      ofs_target<<endl;
    }
  }

  cerr<<"parameter estimation.."<<endl;
  TVector theta(TVector::Zero(NX*NY));
  TReal comp_time_u(GetUserTime()), comp_time_a(GetCurrentTime());
  {
    if(solver==sLeastSquares)
    {
      const TReal lambda(0.1);  // regularization param
      cerr<<"  using regularized least squares"<<endl;
      theta= (Theta.transpose()*Theta+lambda*TMatrix::Identity(GenSize(theta),GenSize(theta))).inverse()*Theta.transpose() * V;
    }
    else if(solver==sSVD)
    {
LERROR("not implemented");
#if 0
      // const double tolerance(1.0e-6);
      const double tolerance(0.01);
      cerr<<"  using SVD"<<endl;
      SVD svd(Theta,SVD::economy);
      DiagMatrix w_inv(svd.singular_values());
      for(int r(w_inv.rows()-1); r>=0; --r)
      {
        if(w_inv(r,r)>tolerance)  w_inv(r,r)= 1.0/w_inv(r,r);
        else  w_inv(r,r)= 0.0;
      }
      // LDBGVAR(w_inv);
      theta= (svd.right_singular_matrix()*w_inv.transpose()*svd.left_singular_matrix().transpose()) * V;
#endif
    }
  }
  comp_time_u= GetUserTime()-comp_time_u;
  comp_time_a= GetCurrentTime()-comp_time_a;

  cerr<<"test (sample).."<<endl;
  TReal rmse_sample(0.0);
  {
    TVector f;
    for(int n(0); n<N_sample; ++n)
    {
      f= Theta.row(n).transpose();
      TReal y= theta.transpose()*f;
      rmse_sample+= Square(y-V(n));
    }
    rmse_sample= real_sqrt(rmse_sample/TReal(N_sample));
  }

  cerr<<"test (gen).."<<endl;
  TReal rmse_gen(0.0);
  {
    ofstream ofs_test("res/test.dat");
    TVector x(TVector::Zero(2));
    TReal n_err(0);
    for(x(0)=-1.3; x(0)<=1.3; x(0)+=0.05)
    {
      for(x(1)=-1.3; x(1)<=1.3; x(1)+=0.05)
      {
        TReal y= theta.transpose()*GetFeature(x);
        ofs_test<<x.transpose()<<"  "<<y<<endl;
        rmse_gen+= Square(y-TargetFunc(x));
        ++n_err;
      }
      ofs_test<<endl;
    }
    rmse_gen= real_sqrt(rmse_gen/n_err);
  }

  cerr<<"done.\n"<<endl;

  cout<<"Result:"<<endl;
  cout<<"  Method: "<<int(solver)<<endl;
  cout<<"  N_feat: "<<GenSize(theta)<<endl;
  cout<<"  N_sample: "<<N_sample<<endl;
  cout<<"  RMSE(gen): "<<rmse_gen<<endl;
  cout<<"  RMSE(sample): "<<rmse_sample<<endl;
  cout<<"  Comp Time(user): "<<comp_time_u<<endl;
  cout<<"  Comp Time(actual): "<<comp_time_a<<endl;

  if(report)
  {
    stringstream filename;
    filename<<"res/res-";
    if(solver==sLeastSquares)  filename<<"LS";
    else if(solver==sSVD)      filename<<"SVD";
    filename<<"-"<<GenSize(theta)<<".dat";
    ofstream ofs_test(filename.str().c_str(), ios::out|ios::app);
    ofs_test<<N_sample<<" "<<rmse_gen<<" "<<rmse_sample<<" "<<comp_time_u<<" "<<comp_time_a<<endl;
  }

  return 0;
}
//-------------------------------------------------------------------------------------------
