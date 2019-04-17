/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2008-2009 Soeren Sonnenburg
 * Copyright (C) 2008-2009 Fraunhofer Institute FIRST and Max Planck Society
 */
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/features/SimpleFeatures.h>
#include <shogun/regression/svr/LibSVR.h>
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

float64_t* lab;
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
  while(getline(ifs,line,'\n'))
  {
    data.push_back(std::list<double>());
    stringstream ss(line);
    while(ss>>value)
    {
      data.back().push_back(value);
    }
  }
}

void load_train_data()
{
  std::list<std::list<double> > fdata, ldata;
  load_data_from_file("../data/toy/fm_train_real.dat", fdata);
  load_data_from_file("../data/toy/label_train_twoclass.dat", ldata);

  assert(fdata.size()==ldata.size());
  assert(ldata.front().size()==1);

  NUM= fdata.size();
  DIMS= fdata.front().size();
  lab= new float64_t[NUM];
  feat= new float64_t[NUM*DIMS];
  int r(0);
  for (std::list<std::list<double> >::const_iterator
      fitr(fdata.begin()),litr(ldata.begin()),flast(fdata.end());
      fitr!=flast; ++fitr,++litr,++r)
  {
    lab[r]= litr->front();
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
  // const float64_t rbf_width=10;
  // const float64_t svm_C=10;
  // const float64_t svm_eps=0.01;
  const float64_t rbf_width=2.1;
  const float64_t svm_C=1.2;
  const float64_t svm_eps=0.0001;
  const float64_t svm_tube_eps=0.01;

  init_shogun(&print_message);

  load_train_data();

  // create train labels
  CLabels* labels=new CLabels();
  labels->set_labels(lab, NUM);
  SG_REF(labels);

  // create train features
  CSimpleFeatures<float64_t>* features = new CSimpleFeatures<float64_t>(feature_cache);
  SG_REF(features);
  features->set_feature_matrix(feat, DIMS, NUM);

  // create gaussian kernel
  CGaussianKernel* kernel = new CGaussianKernel(kernel_cache, rbf_width);
  SG_REF(kernel);
  kernel->init(features, features);

  // create svm via libsvm and train
  CLibSVR* svm = new CLibSVR(svm_C, svm_eps, kernel, labels);
  SG_REF(svm);
//   svm->set_epsilon(svm_eps);
  svm->set_tube_epsilon(svm_tube_eps);
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
      test_feat[(x*test_size+y)*DIMS+0]= 10.0*(double)x/(double)test_size-5.0;
      test_feat[(x*test_size+y)*DIMS+1]= 10.0*(double)y/(double)test_size-5.0;
    }
  }
  CSimpleFeatures<float64_t>* test_features = new CSimpleFeatures<float64_t>(feature_cache);
  SG_REF(test_features);
  test_features->set_feature_matrix(test_feat, DIMS, test_size*test_size);

  // print(kernel->get_num_vec_lhs());
  // print(kernel->get_num_vec_rhs());
  // kernel->init(kernel->get_lhs(), test_features);
  // print(NULL);
  // print(kernel->get_num_vec_lhs());
  // print(kernel->get_num_vec_rhs());
  CLabels* out_labels=svm->classify(test_features);
  print(out_labels->get_num_labels());
  print(kernel->get_num_vec_lhs());
  print(kernel->get_num_vec_rhs());

  // equivalent code:
  // kernel->init(kernel->get_lhs(), test_features);
  // CLabels* out_labels=svm->classify();
  // print(out_labels->get_num_labels());
  // print(kernel->get_num_vec_lhs());
  // print(kernel->get_num_vec_rhs());

  // for (int32_t i=0; i<out_labels->get_num_labels(); i++)
  //   printf("out[%d]=%f\n", i, out_labels->get_label(i));

  ofstream ofsl("dat-rgrs1/labels.dat");
  ofstream ofsol("dat-rgrs1/out_labels.dat");
  ofstream ofsf("dat-rgrs1/features.dat");
  ofstream ofstf("dat-rgrs1/test_features.dat");
  for (int32_t i(0); i<labels->get_num_labels(); ++i)
    ofsl<<labels->get_label(i)<<endl;
  // for (int32_t i(0); i<out_labels->get_num_labels(); ++i)
  //   ofsol<<out_labels->get_label(i)<<endl;
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
      ofsol<<out_labels->get_label(x*test_size+y)<<endl;
      ofstf<<test_features->get_feature_matrix(nf,nv)[(x*test_size+y)*DIMS+0]
           <<" "<<test_features->get_feature_matrix(nf,nv)[(x*test_size+y)*DIMS+1]<<endl;
    }
    ofsol<<endl;
    ofstf<<endl;
  }

  SG_UNREF(labels);
  SG_UNREF(out_labels);
  SG_UNREF(kernel);
  SG_UNREF(features);
  SG_UNREF(test_features);
  SG_UNREF(svm);


  exit_shogun();
  return 0;
}
