#include "dataloader.h"
#include <cstdlib>
#include <cmath>
#include <iostream>

#include <shogun/kernel/GaussianKernel.h>
#include <shogun/features/SimpleFeatures.h>
#include <shogun/regression/svr/LibSVR.h>
#include <shogun/lib/Mathematics.h>
#include <shogun/lib/common.h>
#include <shogun/base/init.h>

using namespace std;
using namespace shogun;

#define print(var) std::cerr<<#var"= "<<(var)<<std::endl

int main(int,char**)
{
  const int32_t feature_cache=0;
  const int32_t kernel_cache=0;
  const float64_t rbf_width=0.1;
  const float64_t svm_C=1.0;
  const float64_t svm_eps=0.0001;
  const float64_t svm_tube_eps=0.01;

  cerr<<"Learning ..."<<endl;
  init_shogun();

  int num,dim;  // number of samples, sample dimension
  float64_t *feat,*lab;
  int tmp1,tmp2;
  LoadDataFromFile("d/input1.dat",feat,num,dim);
  LoadDataFromFile("d/output1.dat",lab,tmp1,tmp2);
  assert(tmp1==num);
  assert(tmp2==1);

  // create training features
  CSimpleFeatures<float64_t> *features;
  features= new CSimpleFeatures<float64_t>(feature_cache);
  features->set_feature_matrix(feat, dim, num);
  SG_REF(features);

  // create gaussian kernel
  CGaussianKernel *kernel;
  kernel= new CGaussianKernel(kernel_cache, rbf_width);
  kernel->init(features, features);
  SG_REF(kernel);

  // create training labels
  CLabels *labels;
  labels= new CLabels();
  labels->set_labels(lab, num);
  SG_REF(labels);

  // create svm via libsvm and train
  CLibSVR *svm;
  svm= new CLibSVR(svm_C, svm_eps, kernel, labels);
  SG_REF(svm);
  svm->set_tube_epsilon(svm_tube_eps);
  svm->train();

  print(svm->get_num_support_vectors());
  print(svm->get_bias());
  cerr<<"Done: Learning"<<endl;


  cerr<<"Testing ..."<<endl;
  // generating testing features
  const int mesh_size(50), test_num(mesh_size*mesh_size);
  const double x1min(-1),x1max(1),x2min(-1),x2max(1);
  float64_t *test_feat= new float64_t[dim*test_num];
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
  test_features->set_feature_matrix(test_feat, dim, test_num);

  // estimating labels
  CLabels* out_labels;
  out_labels= svm->classify(test_features);

  // saving data into file
  for(int x1(0);x1<mesh_size;++x1)
  {
    for(int x2(0);x2<mesh_size;++x2)
    {
      int n= x1*mesh_size+x2;
      cout<<test_feat[dim*n+0]<<" "<<test_feat[dim*n+1]
          <<"   "<<out_labels->get_label(n)<<endl;
    }
    cout<<endl;
  }
  cout<<endl;
  cerr<<"Done: Testing"<<endl;


  SG_UNREF(test_features);
  SG_UNREF(out_labels);

  SG_UNREF(kernel);
  SG_UNREF(features);
  SG_UNREF(labels);
  SG_UNREF(svm);

  return 0;
}
