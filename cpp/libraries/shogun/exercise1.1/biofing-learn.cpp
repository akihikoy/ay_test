#include <bioloid.h>
#include <fstream>
#include <cstdlib>
#include <cmath>

#include "dataloader1.1.h"
#include <iostream>

#include <shogun/kernel/GaussianKernel.h>
#include <shogun/features/SimpleFeatures.h>
#include <shogun/regression/svr/LibSVR.h>
#include <shogun/lib/common.h>
#include <shogun/base/init.h>

using namespace std;
using namespace loco_rabbits;
using namespace serial;
using namespace bioloid;
using namespace shogun;

//-------------------------------------------------------------------------------------------
#define print(var) std::cout<<#var"= "<<(var)<<std::endl
// #define print(var) print_container((var), #var"= ")
// #define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

const int LDIM(1);  // label dimension

int main(int argc, char**argv)
{
  string filename1("biof/fing_param.dat"),filename2("biof/fing_dist.dat");

  const int32_t feature_cache=0;
  const int32_t kernel_cache=0;
  const float64_t rbf_width=3.0;
  const float64_t svm_C=20.0;
  const float64_t svm_eps=0.0001;
  const float64_t svm_tube_eps=0.02;
  int max_sample=-1;

  cerr<<"Learning ..."<<endl;
  init_shogun();

  int num,dim;  // number of samples, sample dimension
  SGVector<float64_t> lab[LDIM];
  SGMatrix<float64_t> feat;
  int tmp1,tmp2;
  feat= LoadMatFromFile(filename1.c_str(),num,dim,max_sample);
  lab[0]= LoadVecFromFile(filename2.c_str(),tmp1,tmp2,0,max_sample);
  assert(tmp1==num);
  // lab[1]= LoadVecFromFile(filename2.c_str(),tmp1,tmp2,1,max_sample);
  // assert(tmp1==num);

  // create train labels
  CLabels *labels[LDIM];
  for(int i(0);i<LDIM;++i)  labels[i]= new CLabels();
  for(int i(0);i<LDIM;++i)  labels[i]->set_labels(lab[i]);
  for(int i(0);i<LDIM;++i)  SG_REF(labels[i]);

  // create train features
  CSimpleFeatures<float64_t> *features;
  features= new CSimpleFeatures<float64_t>(feature_cache);
  features->set_feature_matrix(feat);
  SG_REF(features);

  // create gaussian kernel
  CGaussianKernel *kernel;
  kernel= new CGaussianKernel(kernel_cache, rbf_width);
  kernel->init(features, features);
  SG_REF(kernel);

  // create svm via libsvm and train
  CLibSVR *svm[LDIM];
  for(int i(0);i<LDIM;++i)  svm[i]= new CLibSVR(svm_C, svm_eps, kernel, labels[i]);
  for(int i(0);i<LDIM;++i)  SG_REF(svm[i]);
  for(int i(0);i<LDIM;++i)  svm[i]->set_tube_epsilon(svm_tube_eps);
  for(int i(0);i<LDIM;++i)  svm[i]->train();

  for(int i(0);i<LDIM;++i)  print(svm[i]->get_num_support_vectors());
  for(int i(0);i<LDIM;++i)  print(svm[i]->get_bias());
  // for(int i(0);i<LDIM;++i)  Estimator[i]=svm[i];
  cerr<<"Done: Learning"<<endl;

  // test
  cerr<<"Testing ..."<<endl;
  // generating testing features
  const int mesh_size(50), test_num(mesh_size*mesh_size);
  const double x1min(0),x1max(7),x2min(-2),x2max(3);
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
  ofstream ofs_res("biof/fing_est.dat");
  for(int x1(0);x1<mesh_size;++x1)
  {
    for(int x2(0);x2<mesh_size;++x2)
    {
      int n= x1*mesh_size+x2;
      ofs_res<<test_feat[dim*n+0]<<" "<<test_feat[dim*n+1]
          <<"   "<<out_labels[0]->get_label(n)<<endl;
    }
    ofs_res<<endl;
  }
  ofs_res<<endl;
  ofs_res.close();
  cerr<<"Done: Testing"<<endl;

  #if 0
  ofstream ofs_test("biof/fing_test.dat");
  for(double target(0.1); target<7.0; target+=0.1)
  {
    // get params
    double Param[2]={0.0, 0.0};
    double err(100000.0);
    for(int n(0); n<test_num; ++n)
      if(fabs(out_labels[0]->get_label(n)-target)<err)
      {
        err= fabs(out_labels[0]->get_label(n)-target);
        Param[0]= test_feat[dim*n+0];
        Param[1]= test_feat[dim*n+1];
      }
    ofs_test<<Param[0]<<" "<<Param[1]<<" "<<target<<endl;
  }
  ofs_test.close();
  #endif


  TBioloidController  bioloid;

  bioloid.Connect("/dev/ttyUSB0");
  bioloid.TossMode();
  // // bioloid.TossTest();

  // bioloid.ConnectBS("/dev/ttyUSB0",1);

  int ids[]= {11};
  double angles[SIZE_OF_ARRAY(ids)];
  // double angles_observed[SIZE_OF_ARRAY(ids)];

  LMESSAGE("----");

  // srand((unsigned)time(NULL));

  ofstream ofs_param("biof/fing_param.dat", std::ios::out | std::ios::app);
  ofstream ofs_dist("biof/fing_dist.dat", std::ios::out | std::ios::app);

  while(true)
  {
    double target;

    // swing back
    angles[0]= -90.0;
    bioloid.GoTo(ids,ids+SIZE_OF_ARRAY(ids), angles);
    cout<<" Put the ball on the finger, then type target and press return (-1 to exit) > ";
    cin>>target;
    if(target<0.0)  break;

    // get params
    double Param[2]={0.0, 0.0};
    double err(100000.0);
    for(int n(0); n<test_num; ++n)
      if(fabs(out_labels[0]->get_label(n)-target)<err)
      {
        err= fabs(out_labels[0]->get_label(n)-target);
        Param[0]= test_feat[dim*n+0];
        Param[1]= test_feat[dim*n+1];
      }
    cout<<Param[0]<<" "<<Param[1]<<endl;

    // play swing
    double time_offset(GetCurrentTime());
    while(angles[0]<Param[1]*10.0)
    {
      double t= GetCurrentTime() - time_offset;
      angles[0]= -90.0 + t*180.0*(1.0-cos(Param[0]*t))*0.5;
      // cerr<<t<<"  "<<angles[0]<<endl;
      bioloid.GoTo(ids,ids+SIZE_OF_ARRAY(ids), angles);
      usleep(20000);
    }

    // log result
    cout<<" Measure the ball position (0: not logged) > ";
    double dist(0.0);
    cin>>dist;
    if(dist>0.1)
    {
      cout<<Param[0]<<" "<<Param[1]<<" "<<dist<<endl;
      ofs_param<<Param[0]<<" "<<Param[1]<<endl;
      ofs_dist<<dist<<endl;
    }
  }

  bioloid.Disconnect();

  return 0;
}
//-------------------------------------------------------------------------------------------
