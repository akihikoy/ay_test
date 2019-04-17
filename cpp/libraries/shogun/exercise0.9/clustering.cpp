#include "dataloader.h"
#include <cstdlib>
#include <cmath>
#include <iostream>

#include <shogun/features/SimpleFeatures.h>
#include <shogun/distance/EuclidianDistance.h>
#include <shogun/clustering/KMeans.h>
#include <shogun/lib/Mathematics.h>
#include <shogun/lib/common.h>
#include <shogun/base/init.h>

using namespace std;
using namespace shogun;

#define print(var) std::cerr<<#var"= "<<(var)<<std::endl

int main(int,char**)
{


// traindat = lm.load_numbers('../data/fm_train_real.dat')

// parameter_list = [[traindat,3],[traindat,4]]

// from shogun.Distance import EuclidianDistance
// from shogun.Features import RealFeatures
// from shogun.Clustering import KMeans
// from shogun.Mathematics import Math_init_random
// Math_init_random(17)

  const int32_t feature_cache=0;
  int k(2);

  cerr<<"Learning ..."<<endl;
  init_shogun();

  CMath::init_random();

  int num,dim;  // number of samples, sample dimension
  float64_t *feat;
  LoadDataFromFile("d/input5.dat",feat,num,dim);

  // create training features
  CSimpleFeatures<float64_t> *features;
  features= new CSimpleFeatures<float64_t>(feature_cache);
  features->set_feature_matrix(feat, dim, num);
  SG_REF(features);

  // create distance
  CEuclidianDistance *distance;
  distance= new CEuclidianDistance(features, features);
  SG_REF(distance);

  // create kmeans and train
  CKMeans *kmeans;
  kmeans= new CKMeans(k, distance);
  SG_REF(kmeans);
  kmeans->train();

// feats_train=RealFeatures(fm_train)
// distance=EuclidianDistance(feats_train, feats_train)

// kmeans=KMeans(k, distance)
// kmeans.train()

// out_centers = kmeans.get_cluster_centers()
// kmeans.get_radiuses()
  float64_t *out_centers;
  int tmp1,tmp2;
  kmeans->get_centers(out_centers,tmp1,tmp2);
  print(tmp1);
  print(tmp2);
  cout<<out_centers[0]<<" "<<out_centers[1]<<endl;
  cout<<out_centers[2]<<" "<<out_centers[3]<<endl;

  cerr<<"Done: Learning"<<endl;

// return out_centers, kmeans


#if 0
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
  out_labels= kmeans->apply(test_features);

  // saving data into file
  for(int x1(0);x1<mesh_size;++x1)
  {
    for(int x2(0);x2<mesh_size;++x2)
    {
      int n= x1*mesh_size+x2;
      double label= out_labels->get_label(n);
      cout<<test_feat[dim*n+0]<<" "<<test_feat[dim*n+1]
          <<"   "<<(label>0 ? +1 : -1)<<" "<<label<<endl;
    }
    cout<<endl;
  }
  cout<<endl;
  cerr<<"Done: Testing"<<endl;

  SG_UNREF(test_features);
  SG_UNREF(out_labels);
#endif

  SG_UNREF(distance);
  SG_UNREF(features);
  SG_UNREF(kmeans);

  return 0;
}


