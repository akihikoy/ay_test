#include "simpleode/lib/arm.h"
#include "dataloader.h"
#include <cstdlib>
#include <cmath>

#include <shogun/kernel/GaussianKernel.h>
#include <shogun/features/SimpleFeatures.h>
#include <shogun/classifier/svm/LibSVMOneClass.h>
#include <shogun/lib/Mathematics.h>
#include <shogun/lib/common.h>
#include <shogun/base/init.h>
#include <shogun/lib/AsciiFile.h>

#define print(var) std::cout<<#var"= "<<(var)<<std::endl

using namespace std;
using namespace shogun;
CLibSVMOneClass*Estimator=NULL;

int LDIM(3);  // label (target position) dimension
dVector3 Target={0,0,1};

void CtrlCallback(xode::TEnvironment &env, xode::TDynRobot &robot)
{
  static double target[3]={0,0,0};
  const double kp(1.0);
  for(int j(0);j<3;++j)
    robot.SetAngVelHinge(j,kp*(target[j]-robot.GetAngleHinge(j)));
}

void DrawCallback(xode::TEnvironment &env, xode::TDynRobot &robot)
{
  const int32_t feature_cache=0;
  const int dim(LDIM);
  float64_t *test_feat= new float64_t[dim];
  if(LDIM==2)
  {
    test_feat[0]= Target[0];
    test_feat[1]= Target[2];
  }
  else
  {
    test_feat[0]= Target[0];
    test_feat[1]= Target[1];
    test_feat[2]= Target[2];
  }

  CSimpleFeatures<float64_t> *test_features= new CSimpleFeatures<float64_t>(feature_cache);
  SG_REF(test_features);
  test_features->set_feature_matrix(test_feat, dim, 1);

  CLabels* out_labels;
  out_labels= Estimator->classify(test_features);

  dReal sides[3]={0.3,0.3,0.3};
  dMatrix3 rot={1,0,0,0, 0,1,0,0, 0,0,1,0};
  print(out_labels->get_label(0)-0.5*Estimator->get_bias());
  if(out_labels->get_label(0) > 0.5*Estimator->get_bias())
    dsSetColorAlpha (0.0, 0.0, 1.0, 0.5);
  else
    dsSetColorAlpha (1.0, 0.0, 0.0, 0.5);
  dsDrawBox (Target, rot, sides);

  SG_UNREF(test_features);
  SG_UNREF(out_labels);
}

void KeyEventCallback(xode::TEnvironment &env, xode::TDynRobot &robot, int command)
{
  dReal d(0.05);
  switch(command)
  {
  case 'd': Target[0]+=d; break;
  case 'a': Target[0]-=d; break;
  case 'w': Target[2]+=d; break;
  case 's': Target[2]-=d; break;
  case 'e': Target[1]+=d; break;
  case 'q': Target[1]-=d; break;
  }
}

int main (int argc, char **argv)
{
  int max_sample(-1);
  if(argc>=2)
  {
    stringstream ss(argv[1]);
    ss>> LDIM;
  }
  if(argc>=3)
  {
    stringstream ss(argv[2]);
    ss>> max_sample;
  }

  const int32_t feature_cache=0;
  const int32_t kernel_cache=0;
  const float64_t rbf_width=0.1;
  const float64_t svm_C=1.0;
  const float64_t svm_nu=0.1;
  const float64_t svm_eps=1.0e-5;

  cerr<<"Learning ..."<<endl;
  init_shogun();

  int num,dim;  // number of samples, sample dimension
  float64_t *feat;
  if(LDIM==2)
  {
    LoadDataFromFile("out-2d-pos.dat",feat,num,dim,-1,max_sample);
  }
  else
  {
    LoadDataFromFile("out-3d-pos.dat",feat,num,dim,-1,max_sample);
  }

  // create train features
  CSimpleFeatures<float64_t> *features;
  features= new CSimpleFeatures<float64_t>(feature_cache);
  features->set_feature_matrix(feat, dim, num);
  SG_REF(features);

  // create gaussian kernel
  CGaussianKernel *kernel;
  kernel= new CGaussianKernel(kernel_cache, rbf_width);
  kernel->init(features, features);
  SG_REF(kernel);

  // create svm via libsvm and train
  CLibSVMOneClass *svm;
  svm= new CLibSVMOneClass(svm_C, kernel);
  SG_REF(svm);
  svm->set_epsilon(svm_eps);
  svm->set_nu(svm_nu);
  svm->train();

  print(svm->get_num_support_vectors());
  print(svm->get_bias());
  Estimator=svm;
  cerr<<"Done: Learning"<<endl;

  xode::TEnvironment env;
  dsFunctions fn= xode::SimInit("textures",env);
  xode::ControlCallback= &CtrlCallback;
  xode::DrawCallback= &DrawCallback;
  xode::KeyEventCallback= &KeyEventCallback;

  dsSimulationLoop (argc,argv,400,400,&fn);

  dCloseODE();

  SG_UNREF(kernel);
  SG_UNREF(features);
  SG_UNREF(svm);

  return 0;
}
