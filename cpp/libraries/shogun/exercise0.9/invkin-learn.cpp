#include "simpleode/lib/arm.h"
#include "dataloader.h"
#include <cstdlib>
#include <cmath>

#include <shogun/kernel/GaussianKernel.h>
#include <shogun/features/SimpleFeatures.h>
#include <shogun/regression/svr/LibSVR.h>
#include <shogun/lib/Mathematics.h>
#include <shogun/lib/common.h>
#include <shogun/base/init.h>
#include <shogun/lib/AsciiFile.h>

template<typename T>
inline T URand()
{
  return static_cast<T>(std::rand())/static_cast<T>(RAND_MAX);
}

#define print(var) std::cout<<#var"= "<<(var)<<std::endl

using namespace std;
using namespace shogun;
int LDIM(3);  // label (joint angle) dimension
CLibSVR *Estimator[3]={NULL,NULL,NULL};
dVector3 Target={0,0,1};

void CtrlCallback(xode::TEnvironment &env, xode::TDynRobot &robot)
{
  static int count(0);
  static double jtarget[3]={0,0,0};
  if(count%10==0)
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

    CLabels* out_labels[3];
    for(int i(0);i<LDIM;++i) out_labels[i]= Estimator[i]->classify(test_features);

    if(LDIM==2)  {jtarget[0]=0.0; jtarget[1]=out_labels[0]->get_label(0); jtarget[2]=out_labels[1]->get_label(0);}
    else  {jtarget[0]=out_labels[0]->get_label(0); jtarget[1]=out_labels[1]->get_label(0); jtarget[2]=out_labels[2]->get_label(0);}

    SG_UNREF(test_features);
    for(int i(0);i<LDIM;++i)  SG_UNREF(out_labels[i]);
  }
  ++count;
  const double kp(10.0);
  for(int j(0);j<3;++j)
    robot.SetAngVelHinge(j,kp*(jtarget[j]-robot.GetAngleHinge(j)));
}

void DrawCallback(xode::TEnvironment &env, xode::TDynRobot &robot)
{
  dReal sides[3]={0.3,0.3,0.3};
  dMatrix3 rot={1,0,0,0, 0,1,0,0, 0,0,1,0};
  dsSetColorAlpha (0.0, 0.0, 1.0, 0.5);
  dsDrawBox (Target, rot, sides);
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
  case 'e': if(LDIM==3)Target[1]+=d; break;
  case 'q': if(LDIM==3)Target[1]-=d; break;
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
  const float64_t rbf_width=1.0;
  const float64_t svm_C=1.2;
  const float64_t svm_eps=0.0001;
  const float64_t svm_tube_eps=0.01;

  cerr<<"Learning ..."<<endl;
  init_shogun();

  int num,dim;  // number of samples, sample dimension
  float64_t *feat,*lab[3];
  int tmp1,tmp2;
  if(LDIM==2)
  {
    LoadDataFromFile("out-2d-pos.dat",feat,num,dim,-1,max_sample);
    LoadDataFromFile("out-2d-angle.dat",lab[0],tmp1,tmp2,0,max_sample);
    assert(tmp1==num);
    LoadDataFromFile("out-2d-angle.dat",lab[1],tmp1,tmp2,1,max_sample);
    assert(tmp1==num);
  }
  else
  {
    LoadDataFromFile("out-3d-pos.dat",feat,num,dim,-1,max_sample);
    LoadDataFromFile("out-3d-angle.dat",lab[0],tmp1,tmp2,0,max_sample);
    assert(tmp1==num);
    LoadDataFromFile("out-3d-angle.dat",lab[1],tmp1,tmp2,1,max_sample);
    assert(tmp1==num);
    LoadDataFromFile("out-3d-angle.dat",lab[2],tmp1,tmp2,2,max_sample);
    assert(tmp1==num);
  }

  // create train labels
  CLabels *labels[3];
  for(int i(0);i<LDIM;++i)  labels[i]= new CLabels();
  for(int i(0);i<LDIM;++i)  labels[i]->set_labels(lab[i], num);
  for(int i(0);i<LDIM;++i)  SG_REF(labels[i]);

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
  CLibSVR *svm[3];
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
  xode::ControlCallback= &CtrlCallback;
  xode::DrawCallback= &DrawCallback;
  xode::KeyEventCallback= &KeyEventCallback;

  dsSimulationLoop (argc,argv,400,400,&fn);

  dCloseODE();

  for(int i(0);i<LDIM;++i)  SG_UNREF(labels[i]);
  SG_UNREF(kernel);
  SG_UNREF(features);
  for(int i(0);i<LDIM;++i)  SG_UNREF(svm[i]);

  return 0;
}
