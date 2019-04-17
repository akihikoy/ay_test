#define NOT_MAIN
#include "file-io.cpp"

// 表示用マクロ:
#define print(var)  \
  std::cout<<#var" =\n"<<(var)<<std::endl

using namespace Eigen;
using namespace std;
int main(int argc,char **argv)
{
  if(argc!=2)
  {
    cerr<<"error!\nusage: "<<argv[0]<<" FILE"<<endl;
    return -1;
  }
  MatrixXd samples;
  LoadFromFile(samples,argv[1]);

  // 2. compute the mean and the variance-covariance matrix
  double N(samples.rows());
  // VectorXd mean= samples.colwise().sum()/N;
  VectorXd mean= samples.transpose()*VectorXd::Ones(samples.rows())/N;
  MatrixXd cov= samples.transpose()*samples/N - mean*mean.transpose();
  print(mean);
  print(cov);

  // 3. apply eigen value decomposition
  SelfAdjointEigenSolver<MatrixXd> eig(cov);
  print(eig.eigenvalues());
  print(eig.eigenvectors());

  // 4.1. reduce dimension to 1
  MatrixXd Tr;
  ofstream ofs;
  Tr= eig.eigenvectors().rightCols(1);
  ofs.open("out-red1.dat");
  // ((samples.rowwise()-mean.transpose())*Tr): dim-reduced samples
  // ofs<< (samples*Tr) * Tr.transpose() <<endl;
  ofs<< ( ((samples.rowwise()-mean.transpose())*Tr) * Tr.transpose() ).rowwise()+mean.transpose() <<endl;
  ofs.close();

  // 4.2. reduce dimension to 2
  ofs.open("out-red2.dat");
  Tr= eig.eigenvectors().rightCols(2);
  // ofs<< (samples*Tr) * Tr.transpose() <<endl;
  ofs<< ( ((samples.rowwise()-mean.transpose())*Tr) * Tr.transpose() ).rowwise()+mean.transpose() <<endl;
  ofs.close();

  // 4.3. reduce dimension to 3
  ofs.open("out-red3.dat");
  Tr= eig.eigenvectors().rightCols(3);
  // ofs<< (samples*Tr) * Tr.transpose() <<endl;
  ofs<< ( ((samples.rowwise()-mean.transpose())*Tr) * Tr.transpose() ).rowwise()+mean.transpose() <<endl;
  ofs.close();

  return 0;
}
