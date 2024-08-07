#include "simpleode/lib/arm.h"
#include "dataloader1.1.h"
#include <cstdlib>
#include <cmath>

#include <shogun/kernel/GaussianKernel.h>
#include <shogun/features/SimpleFeatures.h>
#include <shogun/regression/svr/LibSVR.h>
#include <shogun/lib/common.h>
#include <shogun/base/init.h>

template<typename T>
inline T URand()
{
  return static_cast<T>(std::rand())/static_cast<T>(RAND_MAX);
}

#define print(var) std::cout<<#var"= "<<(var)<<std::endl

using namespace std;
using namespace shogun;
int LDIM(3);  // label (target position) dimension
CLibSVR *Estimator[3]={NULL,NULL,NULL};

void CtrlCallback(xode::TEnvironment &env, xode::TDynRobot &robot)
{
  static int count(0);
  static double target[3]={0,0,0};
  if(count%10==0)
  {
    target[0]= (LDIM==2) ? 0.0 : 2.0*(URand<double>()-0.5)*M_PI;
    target[1]= (URand<double>()-0.5)*M_PI;
    target[2]= 1.8*(URand<double>()-0.5)*M_PI;
  }
  ++count;
  const double kp(1.0);
  for(int j(0);j<3;++j)
    robot.SetAngVelHinge(j,kp*(target[j]-robot.GetAngleHinge(j)));
}

void DrawCallback(xode::TEnvironment &env, xode::TDynRobot &robot)
{
  const int32_t feature_cache=0;
  const int dim(LDIM);
  SGMatrix<float64_t> test_feat(dim,1);
  if(LDIM==2)
  {
    test_feat[0]= robot.GetAngleHinge(1);
    test_feat[1]= robot.GetAngleHinge(2);
  }
  else
  {
    test_feat[0]= robot.GetAngleHinge(0);
    test_feat[1]= robot.GetAngleHinge(1);
    test_feat[2]= robot.GetAngleHinge(2);
  }

  CSimpleFeatures<float64_t> *test_features= new CSimpleFeatures<float64_t>(feature_cache);
  SG_REF(test_features);
  test_features->set_feature_matrix(test_feat);

  CLabels* out_labels[3];
  for(int i(0);i<LDIM;++i) out_labels[i]= Estimator[i]->apply(test_features);

  dReal sides[3]={0.3,0.3,0.3};
  dVector3 pos;
  if(LDIM==2)  {pos[0]=out_labels[0]->get_label(0); pos[1]=0.0; pos[2]=out_labels[1]->get_label(0);}
  else  {pos[0]=out_labels[0]->get_label(0); pos[1]=out_labels[1]->get_label(0); pos[2]=out_labels[2]->get_label(0);}
  dMatrix3 rot={1,0,0,0, 0,1,0,0, 0,0,1,0};
  dsSetColorAlpha (0.0, 0.0, 1.0, 0.5);
  dsDrawBox (pos, rot, sides);

  SG_UNREF(test_features);
  for(int i(0);i<LDIM;++i)  SG_UNREF(out_labels[i]);
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
  const float64_t rbf_width=2.1;
  const float64_t svm_C=1.2;
  const float64_t svm_eps=0.0001;
  const float64_t svm_tube_eps=0.01;

  cerr<<"Learning ..."<<endl;
  init_shogun();

  int num,dim;  // number of samples, sample dimension
  SGVector<float64_t> lab[3];
  SGMatrix<float64_t> feat;
  int tmp1,tmp2;
  if(LDIM==2)
  {
    feat= LoadMatFromFile("out-2d-angle.dat",num,dim,max_sample);
    lab[0]= LoadVecFromFile("out-2d-pos.dat",tmp1,tmp2,0,max_sample);
    assert(tmp1==num);
    lab[1]= LoadVecFromFile("out-2d-pos.dat",tmp1,tmp2,1,max_sample);
    assert(tmp1==num);
  }
  else
  {
    feat= LoadMatFromFile("out-3d-angle.dat",num,dim,max_sample);
    lab[0]= LoadVecFromFile("out-3d-pos.dat",tmp1,tmp2,0,max_sample);
    assert(tmp1==num);
    lab[1]= LoadVecFromFile("out-3d-pos.dat",tmp1,tmp2,1,max_sample);
    assert(tmp1==num);
    lab[2]= LoadVecFromFile("out-3d-pos.dat",tmp1,tmp2,2,max_sample);
    assert(tmp1==num);
  }

  // create train labels
  CLabels *labels[3];
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

  dsSimulationLoop (argc,argv,400,400,&fn);

  dCloseODE();

  for(int i(0);i<LDIM;++i)  SG_UNREF(labels[i]);
  SG_UNREF(kernel);
  SG_UNREF(features);
  for(int i(0);i<LDIM;++i)  SG_UNREF(svm[i]);

  return 0;
}
