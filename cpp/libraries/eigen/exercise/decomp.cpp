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
  if(op=="eig")
  {
    if(m.rows()!=m.cols())
      cerr<<"\n###SelfAdjointEigenSolver is only for a self-adjoint matrix"<<endl;
    else
    {
      cout<<"\n###using SelfAdjointEigenSolver (for self-adjoint matrix):"<<endl;
      SelfAdjointEigenSolver<MatrixXd> eig(m);
      MatrixXd D,V;
      print(D=eig.eigenvalues());
      print(V=eig.eigenvectors());
      print(V * D.asDiagonal() * V.transpose());
      print(V * D.asDiagonal() * V.inverse());
    }
    if(m.rows()!=m.cols())
      cerr<<"\n###EigenSolver is only for a square and real matrix"<<endl;
    else
    {
      cout<<"\n###using EigenSolver (for square and real matrix):"<<endl;
      EigenSolver<MatrixXd> eig(m);
      MatrixXcd D,V;
      print(D=eig.eigenvalues());
      print(V=eig.eigenvectors());
      print(V * D.asDiagonal() * V.transpose());
      print(V * D.asDiagonal() * V.inverse());
      print(D.real());  // 複素数から実数を抽出
      print(V.real());  // 複素数から実数を抽出
    }
  }
  else if(op=="svd")
  {
    {
      cout<<"\n###using ComputeThinU | ComputeThinV option:"<<endl;
      JacobiSVD<MatrixXd> svd(m, ComputeThinU | ComputeThinV);
      MatrixXd U,S,V;
      print(S=svd.singularValues());
      print(U=svd.matrixU());
      print(V=svd.matrixV());
      print(U * S.asDiagonal() * V.transpose());
    }
    {
      cout<<"\n###using ComputeFullU | ComputeFullV option:"<<endl;
      JacobiSVD<MatrixXd> svd(m, ComputeFullU | ComputeFullV);
      MatrixXd U,S,V;
      print(S=svd.singularValues());
      print(U=svd.matrixU());
      print(V=svd.matrixV());
      print(U.leftCols(S.size()));
      print(V.leftCols(S.size()));
      print(U.leftCols(S.size()) * S.asDiagonal() * V.leftCols(S.size()).transpose());
    }
  }
  else if(op=="lu")
  {
    {
      cout<<"\n###using FullPivLU:"<<endl;
      FullPivLU<MatrixXd> lu(m);
      int r;
      MatrixXd P,L,Ltmp,U,Q;
      print(P=lu.permutationP());
      print(Q=lu.permutationQ());
      print(lu.matrixLU());
      print(r=std::max(m.rows(),m.cols()));
      print(U=lu.matrixLU().triangularView<Upper>());
      printev(Ltmp= MatrixXd::Identity(r,r));
      printev(Ltmp.block(0,0,m.rows(),m.cols()).triangularView<StrictlyLower>()= lu.matrixLU());
      print(Ltmp);
      print(L=Ltmp.block(0,0,P.cols(),U.rows()));
      print(P.inverse() * L * U * Q.inverse());
    }
    if(m.rows()!=m.cols())
      cerr<<"\n###PartialPivLU is only for a square matrix"<<endl;
    else
    {
      cout<<"\n###using PartialPivLU:"<<endl;
      PartialPivLU<MatrixXd> lu(m);
      int r;
      MatrixXd P,L,Ltmp,U;
      print(P=lu.permutationP());
      print(lu.matrixLU());
      print(r=std::max(m.rows(),m.cols()));
      print(U=lu.matrixLU().triangularView<Upper>());
      printev(Ltmp= MatrixXd::Identity(r,r));
      printev(Ltmp.block(0,0,m.rows(),m.cols()).triangularView<StrictlyLower>()= lu.matrixLU());
      print(Ltmp);
      print(L=Ltmp.block(0,0,P.cols(),U.rows()));
      print(P.inverse() * L * U);
    }
  }
  else if(op=="chol")
  {
    if(m.rows()!=m.cols())
      cerr<<"\n###LLT is only for a square matrix"<<endl;
    else
    {
      cout<<"\n###standard Cholesky decomposition (LLT):"<<endl;
      LLT<MatrixXd> chol(m);
      MatrixXd L,U;
      print(L= chol.matrixL());
      print(chol.matrixLLT());
      print(U= chol.matrixU());
      print(L * L.transpose());
    }
  }
  else
  {
    cerr<<"invalid operator\nselect from eig,svd,lu,chol"<<endl;
    return -1;
  }
  return 0;
}
