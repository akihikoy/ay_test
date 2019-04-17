#include "dataloader1.1.h"
#include <cstdlib>
#include <cmath>
#include <iostream>

#include <shogun/kernel/GaussianKernel.h>
#include <shogun/features/SimpleFeatures.h>
#include <shogun/regression/svr/LibSVR.h>
#include <shogun/lib/common.h>
#include <shogun/base/init.h>

using namespace std;
using namespace shogun;

#define print(var) std::cerr<<#var"= "<<(var)<<std::endl

int main(int,char**)
{
  const int32_t feature_cache=0;
  const int32_t kernel_cache=0;
  const float64_t rbf_width=0.2;
  const float64_t svm_C=1.0;
  const float64_t svm_eps=0.0001;
  const float64_t svm_tube_eps=0.01;

  cerr<<"Learning ..."<<endl;
  init_shogun();

  int num,dim;  // number of samples, sample dimension
  const int LDIM(2);  // label dimension
  SGVector<float64_t> lab[LDIM];
  SGMatrix<float64_t> feat;
  int tmp1,tmp2;
  feat= LoadMatFromFile("d/input2.dat",num,dim);
  lab[0]= LoadVecFromFile("d/output2.dat",tmp1,tmp2,0);
  assert(tmp1==num);
  lab[1]= LoadVecFromFile("d/output2.dat",tmp1,tmp2,1);
  assert(tmp1==num);

  // create training features
  CSimpleFeatures<float64_t> *features;
  features= new CSimpleFeatures<float64_t>(feature_cache);
  features->set_feature_matrix(feat);
  SG_REF(features);

  // create gaussian kernel
  CGaussianKernel *kernel;
  kernel= new CGaussianKernel(kernel_cache, rbf_width);
  kernel->init(features, features);
  SG_REF(kernel);

  // create training labels
  CLabels *labels[LDIM];
  for(int i(0);i<LDIM;++i)  labels[i]= new CLabels();
  for(int i(0);i<LDIM;++i)  labels[i]->set_labels(lab[i]);
  for(int i(0);i<LDIM;++i)  SG_REF(labels[i]);

  // create svm via libsvm and train
  CLibSVR *svm[LDIM];
  for(int i(0);i<LDIM;++i)  svm[i]= new CLibSVR(svm_C, svm_eps, kernel, labels[i]);
  for(int i(0);i<LDIM;++i)  SG_REF(svm[i]);
  for(int i(0);i<LDIM;++i)  svm[i]->set_tube_epsilon(svm_tube_eps);
  for(int i(0);i<LDIM;++i)  svm[i]->train();

  for(int i(0);i<LDIM;++i)  print(svm[i]->get_num_support_vectors());
  for(int i(0);i<LDIM;++i)  print(svm[i]->get_bias());
  cerr<<"Done: Learning"<<endl;


  cerr<<"Testing ..."<<endl;
  // generating testing features
  const int mesh_size(50), test_num(mesh_size*mesh_size);
  const double x1min(-1),x1max(1),x2min(-1),x2max(1);
  SGMatrix<float64_t> test_feat(dim,test_num);
  for(int x1(0);x1<mesh_size;++x1)
    for(int x2(0);x2<mesh_size;++x2)
    {
      int n= x1*mesh_size+x2;
      test_feat[dim*n+0]= x1min+(x1max-x1min)*(double)x1/(double)(mesh_size-1);
      test_feat[dim*n+1]= x2min+(x2max-x2min)*(double)x2/(double)(mesh_size-1);
    }

  // create testing features
  CSimpleFeatures<float64_t> *test_features= new CSimpleFeatures<float64_t>(feature_cache);
  SG_REF(test_features);
  test_features->set_feature_matrix(test_feat);

  // estimating labels
  CLabels* out_labels[LDIM];
  for(int i(0);i<LDIM;++i) out_labels[i]= svm[i]->apply(test_features);

  // saving data into file
  for(int x1(0);x1<mesh_size;++x1)
  {
    for(int x2(0);x2<mesh_size;++x2)
    {
      int n= x1*mesh_size+x2;
      cout<<test_feat[dim*n+0]<<" "<<test_feat[dim*n+1]
          <<"   "<<out_labels[0]->get_label(n)<<" "<<out_labels[1]->get_label(n)<<endl;
    }
    cout<<endl;
  }
  cout<<endl;
  cerr<<"Done: Testing"<<endl;


  SG_UNREF(test_features);
  for(int i(0);i<LDIM;++i)  SG_UNREF(out_labels[i]);

  SG_UNREF(kernel);
  SG_UNREF(features);
  for(int i(0);i<LDIM;++i)  SG_UNREF(labels[i]);
  for(int i(0);i<LDIM;++i)  SG_UNREF(svm[i]);

  return 0;
}
