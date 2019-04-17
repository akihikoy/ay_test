#include <Eigen/Dense>
template <typename t_matrix>
t_matrix PseudoInverse(const t_matrix& m, const double &tolerance=1.e-6)
{
  using namespace Eigen;
  typedef JacobiSVD<t_matrix> TSVD;
  unsigned int svd_opt(ComputeThinU | ComputeThinV);
  if(m.RowsAtCompileTime!=Dynamic || m.ColsAtCompileTime!=Dynamic)
    svd_opt= ComputeFullU | ComputeFullV;
  TSVD svd(m, svd_opt);
  const typename TSVD::SingularValuesType &sigma(svd.singularValues());
  typename TSVD::SingularValuesType sigma_inv(sigma.size());

  for(long i=0; i<sigma.size(); ++i)
  {
    if(sigma(i) > tolerance)
      sigma_inv(i)= 1.0/sigma(i);
    else
      sigma_inv(i)= 0.0;
  }
  return svd.matrixV()*sigma_inv.asDiagonal()*svd.matrixU().transpose();
}

#ifndef NOT_MAIN
#define NOT_MAIN
#include "file-io.cpp"

// 表示用マクロ:
#define print(var)  \
  std::cout<<#var"= "<<std::endl<<(var)<<std::endl

using namespace Eigen;
using namespace std;
int main(int argc,char **argv)
{
  if(argc!=2)
  {
    cerr<<"error!\nusage: "<<argv[0]<<" FILE"<<endl;
    return -1;
  }
  MatrixXd m;
  LoadFromFile(m,argv[1]);

  print(m);
  if(m.rows()==m.cols())  print(m.inverse());
  if(m.rows()==m.cols())  print(m.inverse()*m);
  print(PseudoInverse(m));
  print(PseudoInverse(m)*m);
  print(m*PseudoInverse(m));
  return 0;
}
#endif // NOT_MAIN
