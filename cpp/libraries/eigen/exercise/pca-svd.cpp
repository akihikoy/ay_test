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

  // 3. apply singular value decomposition
  JacobiSVD<MatrixXd> svd(cov, ComputeThinU | ComputeThinV);
  print(svd.singularValues());
  print(svd.matrixU());
  print(svd.matrixV());

  // 4.1. reduce dimension to 1
  MatrixXd Tr;
  ofstream ofs;
  Tr= svd.matrixV().leftCols(1);
  ofs.open("out-red1.dat");
  // (samples*Tr): dim-reduced samples
  ofs<< (samples*Tr) * Tr.transpose() <<endl; // remap to the original space
  ofs.close();

  // 4.2. reduce dimension to 2
  ofs.open("out-red2.dat");
  Tr= svd.matrixV().leftCols(2);
  ofs<< (samples*Tr) * Tr.transpose() <<endl;
  ofs.close();

  return 0;
}
