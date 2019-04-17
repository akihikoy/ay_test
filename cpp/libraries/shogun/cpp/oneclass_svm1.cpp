#include <shogun/kernel/GaussianKernel.h>
#include <shogun/features/SimpleFeatures.h>
// #include <shogun/classifier/svm/LibSVM.h>
#include <shogun/classifier/svm/LibSVMOneClass.h>
#include <shogun/lib/Mathematics.h>
#include <shogun/lib/common.h>
#include <shogun/base/init.h>
#include <shogun/lib/AsciiFile.h>

#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <fstream>
#include <sstream>
#include <list>

#define print(var) std::cout<<#var"= "<<(var)<<std::endl

using namespace shogun;
using namespace std;

// #define NUM 100
// #define DIMS 2
// #define DIST 0.5

float64_t* feat;
int NUM,DIMS;

void print_message(FILE* target, const char* str)
{
  fprintf(target, "%s", str);
}

void load_data_from_file(const char filename[], std::list<std::list<double> > &data)
{
  using namespace std;
  ifstream ifs(filename);
  string line;
  double value;
  int r(0);
  while(getline(ifs,line,'\n'))
  {
    data.push_back(std::list<double>());
    stringstream ss(line);
    while(ss>>value)
    {
      data.back().push_back(value);
    }
    ++r;
    // if(r>=500) return;
  }
}

void load_train_data()
{
  std::list<std::list<double> > fdata;
  // load_data_from_file("../data/toy/fm_train_real.dat", fdata);
  load_data_from_file("../../../collabo/it3/2011bc09-ml/ex-shogun/out-2d-pos.dat", fdata);

  NUM= fdata.size();
  DIMS= fdata.front().size();
  feat= new float64_t[NUM*DIMS];
  int r(0);
  for (std::list<std::list<double> >::const_iterator
      fitr(fdata.begin()),flast(fdata.end());
      fitr!=flast; ++fitr,++r)
  {
    int c(0);
    for (std::list<double>::const_iterator ditr(fitr->begin()),dlast(fitr->end());
        ditr!=dlast; ++ditr,++c)
    {
      feat[r*DIMS+c]= *ditr;
    }
  }
}

int main()
{
  const int32_t feature_cache=0;
  const int32_t kernel_cache=0;
  // const float64_t rbf_width=0.5;
  const float64_t rbf_width=0.1;
  const float64_t svm_C=1.0;
  const float64_t svm_nu=0.1;
  const float64_t svm_eps=1.0e-5;

  init_shogun(&print_message);

  load_train_data();
  print(NUM);
  print(DIMS);

  // create train features
  CSimpleFeatures<float64_t>* features = new CSimpleFeatures<float64_t>(feature_cache);
  SG_REF(features);
  features->set_feature_matrix(feat, DIMS, NUM);

  // create gaussian kernel
  CGaussianKernel* kernel = new CGaussianKernel(kernel_cache, rbf_width);
  SG_REF(kernel);
  kernel->init(features, features);

  // create svm via libsvm and train
  CLibSVMOneClass* svm = new CLibSVMOneClass(svm_C, kernel);
  SG_REF(svm);
  svm->set_epsilon(svm_eps);
  svm->set_nu(svm_nu);
  // svm->set_bias_enabled(false); // NOTE: IMPORTANT!!
  svm->train();

  print(svm->get_num_support_vectors());
  print(svm->get_bias());

/*
  TODO:
    Save support vectors and draw them.
*/

  // create test features
  int test_size(50);
  float64_t *test_feat= new float64_t[test_size*test_size*DIMS];
  for (int x(0);x<test_size;++x)
  {
    for (int y(0);y<test_size;++y)
    {
      test_feat[(x*test_size+y)*DIMS+0]= 3.0*((double)x/(double)test_size-0.5);
      test_feat[(x*test_size+y)*DIMS+1]= 3.0*((double)y/(double)test_size-0.5);
    }
  }
  CSimpleFeatures<float64_t>* test_features = new CSimpleFeatures<float64_t>(feature_cache);
  SG_REF(test_features);
  test_features->set_feature_matrix(test_feat, DIMS, test_size*test_size);

  CLabels* out_labels=svm->classify(test_features);
  print(out_labels->get_num_labels());
  print(kernel->get_num_vec_lhs());
  print(kernel->get_num_vec_rhs());

  ofstream ofsol("dat-1cls1/out_labels.dat");
  ofstream ofsf("dat-1cls1/features.dat");
  ofstream ofstf("dat-1cls1/test_features.dat");
  int nf,nv;
  for (int32_t i(0); i<NUM; ++i)
    ofsf<<features->get_feature_matrix(nf,nv)[i*DIMS]
        <<" "<<features->get_feature_matrix(nf,nv)[i*DIMS+1]<<endl;
  // for (int32_t i(0); i<test_size*test_size; ++i)
  //   ofstf<<test_features->get_feature_matrix(nf,nv)[i*DIMS]
  //        <<" "<<test_features->get_feature_matrix(nf,nv)[i*DIMS+1]<<endl;
  for (int x(0);x<test_size;++x)
  {
    for (int y(0);y<test_size;++y)
    {
      // ofsol<<out_labels->get_label(x*test_size+y)<<endl;
      ofsol<<out_labels->get_label(x*test_size+y) - 0.5*svm->get_bias()<<endl;
      ofstf<<test_features->get_feature_matrix(nf,nv)[(x*test_size+y)*DIMS+0]
           <<" "<<test_features->get_feature_matrix(nf,nv)[(x*test_size+y)*DIMS+1]<<endl;
    }
    ofsol<<endl;
    ofstf<<endl;
  }

  SG_UNREF(out_labels);
  SG_UNREF(kernel);
  SG_UNREF(features);
  SG_UNREF(test_features);
  SG_UNREF(svm);


  exit_shogun();
  return 0;
}
