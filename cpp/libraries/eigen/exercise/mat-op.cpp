#define NOT_MAIN
#include "file-io.cpp"

// 表示用マクロ:
#define print(var)  \
  std::cout<<#var" =\n"<<(var)<<std::endl
#define printev(var)  \
  do{std::cout<<#var<<std::endl;var;}while(false)

using namespace Eigen;
using namespace std;
int main(int argc,char **argv)
{
  if(argc!=3)
  {
    cerr<<"error!\nusage: "<<argv[0]<<" FILE OP"<<endl;
    return -1;
  }
  MatrixXd m;
  LoadFromFile(m,argv[1]);
  print(m);

  string op(argv[2]);
  if(op=="t")
  {
    print(m.transpose());
  }
  else if(op=="tri")
  {
    MatrixXd m2;
    print(m2=m.triangularView<Upper>());
    print(m2=m.triangularView<Lower>());
    print(m2=m.triangularView<StrictlyUpper>());
    print(m2=m.triangularView<StrictlyLower>());
    print(m2=m.triangularView<UnitUpper>());
    print(m2=m.triangularView<UnitLower>());
  }
  else if(op=="diag")
  {
    print(m.diagonal());
    printev(m.diagonal()=10.0*VectorXd::Ones(m.diagonalSize()));
    print(m);
  }
  else
  {
    cerr<<"invalid operator\nselect from t,tri,diag"<<endl;
    return -1;
  }
  return 0;
}
