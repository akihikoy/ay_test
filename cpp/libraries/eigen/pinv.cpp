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

#include <iostream>
using namespace Eigen;
using namespace std;

int main(int,char**)
{
  MatrixXd m(2,3);
  m<< 1,2,3,
      3,2,1;
  cout<<"m= "<<endl<<m<<endl;
  cout<<"PseudoInverse(m)= "<<endl<<PseudoInverse(m)<<endl;
  cout<<"m*PseudoInverse(m)= "<<endl<<m*PseudoInverse(m)<<endl;
  return 0;
}
