#include "simpleode/lib/fing.h"
#include "dataloader1.1.h"
#include <iostream>
#include <fstream>

#include <shogun/kernel/GaussianKernel.h>
#include <shogun/features/SimpleFeatures.h>
#include <shogun/regression/svr/LibSVR.h>
#include <shogun/lib/common.h>
#include <shogun/base/init.h>

#define print(var) std::cout<<#var"= "<<(var)<<std::endl

using namespace std;
using namespace shogun;

const int LDIM(2);  // label dimension
CLibSVR *Estimator[2]={NULL,NULL};

double Params[2]={3.0,2.0};
double Target(1.0);

void set_params(const double &target)
{
  const int32_t feature_cache=0;
  const int dim(1);
  SGMatrix<float64_t> test_feat(dim,1);

  test_feat[0]= target;

  CSimpleFeatures<float64_t> *test_features= new CSimpleFeatures<float64_t>(feature_cache);
  SG_REF(test_features);
  test_features->set_feature_matrix(test_feat);

  CLabels* out_labels[LDIM];
  for(int i(0);i<LDIM;++i) out_labels[i]= Estimator[i]->apply(test_features);

  Params[0]= out_labels[0]->get_label(0);
  Params[1]= out_labels[1]->get_label(0);

  SG_UNREF(test_features);
  for(int i(0);i<LDIM;++i)  SG_UNREF(out_labels[i]);
}

void test_ctrl(xode::TEnvironment &env, xode::TDynRobot &robot)
{
  if(env.Time()>3.0 && robot.GetBodyContact(3))
  {
    env.Reset();
    set_params(Target);
    return;
  }

  dReal target= Params[0]*std::sin(Params[1]*env.Time());

  const double kp(3.0);
  robot.SetAngVelHinge(0,kp*(target-robot.GetAngleHinge(0)));
}

void test_draw(xode::TEnvironment &env, xode::TDynRobot &robot)
{
  dReal sides[3]={0.3,0.3,0.05};
  dVector3 pos={Target,0,0};
  dMatrix3 rot={1,0,0,0, 0,1,0,0, 0,0,1,0};
  dsSetColorAlpha (0.0, 0.0, 1.0, 0.5);
  dsDrawBox (pos, rot, sides);
}

void test_keyevent(xode::TEnvironment &env, xode::TDynRobot &robot, int command)
{
  if(command=='r')  env.Reset();
  else if(command=='x')  Target-=0.1;
  else if(command=='z')  Target+=0.1;
  cerr<<"Target: "<<Target<<endl;
}

int main (int argc, char **argv)
{
  int max_sample(-1);

  string filename1("out-pos.dat"),filename2("out-params.dat");
  if(argc>1)  filename1= argv[1];
  if(argc>2)  filename2= argv[2];

  const int32_t feature_cache=0;
  const int32_t kernel_cache=0;
  const float64_t rbf_width=2.1;
  const float64_t svm_C=1.2;
  const float64_t svm_eps=0.0001;
  const float64_t svm_tube_eps=0.01;

  cerr<<"Learning ..."<<endl;
  init_shogun();

  int num,dim;  // number of samples, sample dimension
  SGVector<float64_t> lab[LDIM];
  SGMatrix<float64_t> feat;
  int tmp1,tmp2;
  feat= LoadMatFromFile(filename1.c_str(),num,dim,max_sample);
  lab[0]= LoadVecFromFile(filename2.c_str(),tmp1,tmp2,0,max_sample);
  assert(tmp1==num);
  lab[1]= LoadVecFromFile(filename2.c_str(),tmp1,tmp2,1,max_sample);
  assert(tmp1==num);

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
  for(int i(0);i<LDIM;++i)  Estimator[i]=svm[i];
  cerr<<"Done: Learning"<<endl;


  xode::TEnvironment env;
  dsFunctions fn= xode::SimInit("textures",env);
  xode::ControlCallback= &test_ctrl;
  xode::DrawCallback= &test_draw;
  xode::KeyEventCallback= &test_keyevent;

  cerr<<"Push \'X\' to move target in -x direction"<<endl;
  cerr<<"Push \'Z\' to move target in +x direction"<<endl;
  cerr<<"Push \'R\' to reset the simpulator"<<endl;

  set_params(Target);
  dsSimulationLoop (argc,argv,400,400,&fn);

  dCloseODE();


  for(int i(0);i<LDIM;++i)  SG_UNREF(labels[i]);
  SG_UNREF(kernel);
  SG_UNREF(features);
  for(int i(0);i<LDIM;++i)  SG_UNREF(svm[i]);

  return 0;
}
