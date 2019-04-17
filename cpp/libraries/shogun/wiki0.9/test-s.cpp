#include <shogun/kernel/GaussianKernel.h>
#include <shogun/features/SimpleFeatures.h>
#include <shogun/regression/svr/LibSVR.h>
#include <shogun/lib/Mathematics.h>
#include <shogun/lib/common.h>
#include <shogun/base/init.h>
#include <shogun/lib/AsciiFile.h>
using namespace shogun;
using namespace std;
int main()
{
  init_shogun();

  const int NUM(10),DIMS(1);
  float64_t* lab= new float64_t[NUM];
  float64_t* feat= new float64_t[NUM*DIMS];
  for (int x(0);x<NUM;++x)
  {
    lab[x]= (double)x/20.0;
    feat[x*DIMS]= (double)x/10.0;
  }

  CLabels* labels=new CLabels();
  labels->set_labels(lab, NUM);
  SG_REF(labels);

  CSimpleFeatures<float64_t>* features = new CSimpleFeatures<float64_t>(0);
  SG_REF(features);
  features->set_feature_matrix(feat, DIMS, NUM);

  CGaussianKernel* kernel = new CGaussianKernel(0, /*rbf_width*/2.0);
  SG_REF(kernel);
  kernel->init(features, features);

  CLibSVR* svm = new CLibSVR(/*C*/1.2, /*eps*/0.001, kernel, labels);
  SG_REF(svm);
  svm->set_tube_epsilon(0.01);
  svm->train();

  cout<<"num of support vectors: "<<svm->get_num_support_vectors()<<endl;

  SG_UNREF(labels);
  SG_UNREF(kernel);
  SG_UNREF(features);
  SG_UNREF(svm);

  exit_shogun();
  return 0;
}
