#include <cmath>
#include <iostream>
#include <fstream>

#include <shogun/kernel/GaussianKernel.h>
#include <shogun/features/SimpleFeatures.h>
#include <shogun/classifier/svm/LibSVM.h>
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
  const float64_t rbf_width=0.5;
  const float64_t svm_C=1.0;
  const float64_t svm_eps=0.0001;

  cerr<<"Learning ..."<<endl;
  init_shogun();

  // generate training data
  ofstream ofs_in("in.dat");
  int num(100),dim(1);  // number of samples, sample dimension
  float64_t *feat,*lab;
  feat= new float64_t[num*dim];
  lab= new float64_t[num];
  for(int i(0);i<num;++i)
  {
    feat[dim*i]= ((double)i/(double)num-0.5)*6.0;
    lab[i]= (sin(2.0*feat[dim*i])>0 ? +1 : -1);
    ofs_in<<feat[dim*i]<<" "<<lab[i]<<endl;
  }

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
  CLibSVM *svm;
  svm= new CLibSVM(svm_C, kernel, labels);
  SG_REF(svm);
  svm->set_epsilon(svm_eps);
  svm->train();

  print(svm->get_num_support_vectors());
  print(svm->get_bias());
  cerr<<"Done: Learning"<<endl;


  cerr<<"Testing ..."<<endl;
  // generating testing features
  const int mesh_size(200), test_num(mesh_size);
  const double xmin(-4),xmax(4);
  float64_t *test_feat= new float64_t[dim*test_num];
  for(int x(0);x<mesh_size;++x)
    test_feat[dim*x]= xmin+(xmax-xmin)*(double)x/(double)(mesh_size-1);

  // create testing features
  CSimpleFeatures<float64_t> *test_features= new CSimpleFeatures<float64_t>(feature_cache);
  SG_REF(test_features);
  test_features->set_feature_matrix(test_feat, dim, test_num);

  // estimating labels
  CLabels* out_labels;
  out_labels= svm->classify(test_features);

  // saving data into file
  ofstream ofs_out("out.dat");
  for(int x(0);x<mesh_size;++x)
  {
    double label= out_labels->get_label(x);
    ofs_out<<test_feat[dim*x]<<"   "<<(label>0 ? +1 : -1)<<" "<<label<<endl;
  }
  cerr<<"Done: Testing"<<endl;


  SG_UNREF(test_features);
  SG_UNREF(out_labels);

  SG_UNREF(kernel);
  SG_UNREF(features);
  SG_UNREF(labels);
  SG_UNREF(svm);

  return 0;
}
