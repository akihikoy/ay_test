//-------------------------------------------------------------------------------------------
/*! \file    ros_plane_seg1.cpp
    \brief   Plane segmentation ported from ../python/plane_seg1.py
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Jan.13, 2021

g++ -O2 -g -W -Wall -o ros_plane_seg1.out ros_plane_seg1.cpp  -I../include -I/opt/ros/kinetic/include -pthread -llog4cxx -lpthread -L/opt/ros/kinetic/lib -rdynamic -lroscpp -lrosconsole -lroscpp_serialization -lrostime -lcv_bridge -lopencv_highgui -lopencv_imgproc -lopencv_core -Wl,-rpath,/opt/ros/kinetic/lib

*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <set>
#include <vector>
#include <list>
#include <map>
#include <cstdio>
// #include <cstdlib>
#include <sys/time.h>  // gettimeofday
//-------------------------------------------------------------------------------------------

inline double GetCurrentTime(void)
{
  struct timeval time;
  gettimeofday (&time, NULL);
  return static_cast<double>(time.tv_sec) + static_cast<double>(time.tv_usec)*1.0e-6;
  // return ros::Time::now().toSec();
}
//-------------------------------------------------------------------------------------------

/*

inline unsigned Srand(void)
{
  unsigned seed ((unsigned)time(NULL));
  srand(seed);
  return seed;
}
//-------------------------------------------------------------------------------------------

inline TReal Rand (const double &max)
{
  return (max)*static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
}
//-------------------------------------------------------------------------------------------

inline TReal Rand (const double &min, const double &max)
{
  return Rand(max - min) + min;
}
//-------------------------------------------------------------------------------------------

inline TReal Rand (const long double &max)
{
  return (max)*static_cast<long double>(rand()) / static_cast<long double>(RAND_MAX);
}
//-------------------------------------------------------------------------------------------

inline TReal Rand (const long double &min, const long double &max)
{
  return Rand(max - min) + min;
}
//-------------------------------------------------------------------------------------------

inline int Rand (int min, int max)
  // return [min,max]
{
  // return static_cast<int>(Rand(static_cast<double>(min),static_cast<double>(max+1)));
  return static_cast<int>(real_floor(Rand(static_cast<double>(min),static_cast<double>(max+1))));
}
//-------------------------------------------------------------------------------------------

*/


// Feature definition for clustering (interface class).
class TImgPatchFeatIF
{
public:
  // Set up the feature.  Parameters may be added.
  TImgPatchFeatIF()  {}
  // Get a feature vector for an image patch img_patch.
  // Return cv::Mat() if it is impossible to get a feature.
  virtual cv::Mat Feat(const cv::Mat &img_patch) const = 0;
  // Get a difference (scholar value) between two features.
  virtual double Diff(const cv::Mat &f1, const cv::Mat &f2) const = 0;
  // Compute a weighted sum of two features. i.e. w1*f1 + (1.0-w1)*f2
  virtual cv::Mat WSum(const cv::Mat &f1, const cv::Mat &f2, const double &w1) const = 0;
};
//-------------------------------------------------------------------------------------------

//Feature of the normal of a patch.
//  th_plane: The feature is None if the normal length is greater than this value (smaller value is more like plane).
//  step: Interval of points for calculation.
class TImgPatchFeatNormal : public TImgPatchFeatIF
{
public:
  TImgPatchFeatNormal(const double &th_plane=0.4, int step=1)
    {
      th_plane_= th_plane;
      step_= step;
    }
  // Get a normal vector of an image patch img_patch.
  cv::Mat Feat(const cv::Mat &img_patch) const
    {
      // Extract effective depth points.
      int num_data(0);
      for(int r(0);r<img_patch.rows;r+=step_)
        for(int c(0);c<img_patch.cols;c+=step_)
          if(img_patch.at<double>(r,c)>0)  ++num_data;
      if(num_data<3)  return cv::Mat();
      cv::Mat points(num_data,3,CV_64F);
      for(int r(0),i(0);r<img_patch.rows;r+=step_)
        for(int c(0);c<img_patch.cols;c+=step_)
        {
          const double &h= img_patch.at<double>(r,c);
          if(h>0)
          {
            points.at<double>(i,0)= c;
            points.at<double>(i,1)= r;
            points.at<double>(i,2)= h;
            ++i;
          }
        }

      cv::PCA pca(points, cv::Mat(), CV_PCA_DATA_AS_ROW);
      cv::Mat normal= pca.eigenvectors.row(2);
      if(normal.at<double>(0,2)<0)  normal= -normal;
      if(pca.eigenvalues.at<double>(2) > th_plane_)  return cv::Mat();

      return normal;
    }
  // Get an angle [0,pi] between two features.
  double Diff(const cv::Mat &f1, const cv::Mat &f2) const
    {
      double cos_th= f1.dot(f2) / (cv::norm(f1)*cv::norm(f2));
      if(cos_th>1.0)  cos_th= 1.0;
      else if(cos_th<-1.0)  cos_th= -1.0;
      return std::acos(cos_th);
    }
  // Compute a weighted sum of two features. i.e. w1*f1 + (1.0-w1)*f2
  cv::Mat WSum(const cv::Mat &f1, const cv::Mat &f2, const double &w1) const
    {
      cv::Mat ws= w1*f1 + (1.0-w1)*f2;
      double ws_norm= cv::norm(ws);
      if(ws_norm<1.0e-6)
        CV_Error(CV_StsError, "TImgPatchFeatNormal: Computing WSum for normals of opposite directions.");
      return ws/ws_norm;
    }
private:
  double th_plane_;
  int step_;
};
//-------------------------------------------------------------------------------------------

/*
#Feature of the average depth of a patch.
class TImgPatchFeatAvrDepth(TImgPatchFeatIF):
  def __init__(self):
    pass
  #Get an average depth of an image patch img_patch.
  def __call__(self,img_patch):
    img_valid= img_patch[img_patch!=0]
    if len(img_valid)==0:    return None
    return np.array([np.mean(img_valid)])
  '''
  #Equal to the above __call__ (for test).
  def __call__(self,img_patch):
    h,w= img_patch.shape[:2]
    #points= [[x-w/2,y-h/2,img_patch[y,x]] for y in range(h) for x in range(w) if img_patch[y,x]!=0]
    points= np.vstack([np.where(img_patch!=0), img_patch[img_patch!=0].ravel()]).T[:,[1,0,2]] - [w/2,h/2,0]
    if len(points)==0:  return None
    return np.array([np.mean(points,axis=0)[-1]])
  '''
  #Get a difference (scholar value) between two features.
  def Diff(self,f1,f2):
    return np.linalg.norm(f1-f2)
  #Compute a weighted sum of two features. i.e. w1*f1 + (1.0-w1)*f2
  def WSum(self,f1,f2,w1):
    return w1*f1 + (1.0-w1)*f2
//-------------------------------------------------------------------------------------------
*/

struct TClusterNode
{
  std::vector<cv::Rect> patches;
  cv::Mat feat;
  std::set<int> neighbors;  // Neighbor nodes that are candidate to be merged.

  TClusterNode()  {}
  TClusterNode(const cv::Rect &patch0, const cv::Mat feat0)
    {
      patches.push_back(patch0);
      feat= feat0;
    }
};

// Segment img by a given feature model such as normal.
std::list<TClusterNode> ClusteringByFeatures(const cv::Mat &img, int w_patch, const TImgPatchFeatIF &f_feat, const double &th_feat=15.0)
{
  // img= img.reshape((img.shape[0],img.shape[1]))

  /*
  depth image --> patches (w_patch x w_patch); each patch has a feat.
  patches --> nodes = initial planes
  nodes= [TClusterNode([(x,y,w_patch,w_patch)],f_feat(img[y:y+w_patch,x:x+w_patch]))
          for y in range(0,img.shape[0],w_patch)
          for x in range(0,img.shape[1],w_patch)]
  */
  int Nu= img.cols/w_patch;
  int Nv= img.rows/w_patch;
  /*DEBUG*/double debug_t_start= GetCurrentTime();
  std::vector<TClusterNode> nodes;
  typedef std::pair<int,int> TLoc;  // location.
  typedef int TNodeItr;
  std::map<TLoc, TNodeItr> node_map;
  for(int v(0);v<Nv;++v)
    for(int u(0);u<Nu;++u)
    {
      cv::Rect patch(u*w_patch,v*w_patch,w_patch,w_patch);
      cv::Mat feat= f_feat.Feat(img(patch));
      nodes.push_back(TClusterNode(patch,feat));
//       TNodeItr itr= nodes.end();
//       --itr;
      node_map[TLoc(u,v)]= nodes.size()-1;
    }
  /*DEBUG*/std::cerr<<"DEBUG:feat cmp time: "<<GetCurrentTime()-debug_t_start<<std::endl;
  /*DEBUG*/std::cerr<<"DEBUG:Nu,Nv: "<<Nu<<", "<<Nv<<std::endl;
  // dneighbors= ((1,1),(1,0),(1,-1),(0,1),(0,-1),(-1,1),(-1,0),(-1,-1))
  int dneighbors[][2]= {{1,0},{0,1},{0,-1},{-1,0}};
  for(int v(0);v<Nv;++v)
    for(int u(0);u<Nu;++u)
    {
      TNodeItr inode= node_map[TLoc(u,v)];
      if(nodes[inode].feat.empty())  continue;
      for(int inei(0);inei<4;++inei)
      {
        int du(dneighbors[inei][0]), dv(dneighbors[inei][1]);
        if(0<=u+du && u+du<Nu && 0<=v+dv && v+dv<Nv
                && !nodes[node_map[TLoc(u+du,v+dv)]].feat.empty())
          nodes[inode].neighbors.insert(node_map[TLoc(u+du,v+dv)]);
      }
    }
  // nodes= filter(lambda node:node.feat is not None, nodes)
//   for(TNodeItr itr(nodes.end()); itr!=nodes.begin();)
//   {
//     --itr;
//     if(itr->feat.empty())  itr= a.erase(itr);
//   }

  // Clustering nodes.
  std::list<TNodeItr> inodes, iclusters;
  for(TNodeItr itr(0),itr_end(nodes.size()); itr!=itr_end; ++itr)
    if(!nodes[itr].feat.empty())  inodes.push_back(itr);
  while(inodes.empty())
  {
    TNodeItr inode= inodes.front();  inodes.pop_front();  //FIXME: pop a random element?
    std::set<TNodeItr> neighbors= nodes[inode].neighbors;
    nodes[inode].neighbors= std::set<TNodeItr>();
    for(std::set<TNodeItr>::iterator iinode2(neighbors.begin()),iinode2_end(neighbors.end()); iinode2!=iinode2_end; ++iinode2)
    {
      TNodeItr inode2(*iinode2);
      nodes[inode2].neighbors.erase(inode);
      if(f_feat.Diff(nodes[inode].feat,nodes[inode2].feat) < th_feat)
      {
        // Merge inode2 into inode:
        double r1= double(nodes[inode].patches.size())/double(nodes[inode].patches.size()+nodes[inode2].patches.size());
        nodes[inode].feat= f_feat.WSum(nodes[inode].feat, nodes[inode2].feat, r1);
        nodes[inode].patches.insert(nodes[inode].patches.end(), nodes[inode2].patches.begin(), nodes[inode2].patches.end());
        // nodes[inode].neighbors.update(nodes[inode2].neighbors-neighbors)
        std::set<TNodeItr> n_diff;
        std::set_difference(nodes[inode2].neighbors.begin(),nodes[inode2].neighbors.end(), neighbors.begin(),neighbors.end(), std::inserter(n_diff,n_diff.begin()));
        nodes[inode].neighbors.insert(n_diff.begin(),n_diff.end());
        if(std::find(inodes.begin(),inodes.end(),inode2)!=inodes.end())  inodes.remove(inode2);
        for(std::set<TNodeItr>::iterator iinode3(nodes[inode2].neighbors.begin()),iinode3_end(nodes[inode2].neighbors.end()); iinode3!=iinode3_end; ++iinode3)
          nodes[*iinode3].neighbors.erase(inode2);
        n_diff.clear();
        std::set_difference(nodes[inode2].neighbors.begin(),nodes[inode2].neighbors.end(), neighbors.begin(),neighbors.end(), std::inserter(n_diff,n_diff.begin()));
        for(std::set<TNodeItr>::iterator iinode3(n_diff.begin()),iinode3_end(n_diff.end()); iinode3!=iinode3_end; ++iinode3)
          nodes[*iinode3].neighbors.insert(inode);
      }
    }
    if(!nodes[inode].neighbors.empty())
      inodes.push_back(inode);
    else
      iclusters.push_back(inode);
  }
  std::list<TClusterNode> clusters;
  for(std::list<TNodeItr>::iterator itr(inodes.begin()),itr_end(inodes.end()); itr!=itr_end; ++itr)
    clusters.push_back(nodes[*itr]);
  return clusters;
}
//-------------------------------------------------------------------------------------------


/*
#Convert patch points to a binary image.
def PatchPointsToImg(patches):
  if len(patches)==0:  return None,None,None,None
  _,_,patch_w,patch_h= patches[0]
  patches_pts= np.array([[y/patch_h,x/patch_w] for x,y,_,_ in patches])
  patches_topleft= np.min(patches_pts,axis=0)
  patches_btmright= np.max(patches_pts,axis=0)
  patches_img= np.zeros(patches_btmright-patches_topleft+[1,1])
  patches_pts_o= patches_pts - patches_topleft
  patches_img[patches_pts_o[:,0],patches_pts_o[:,1]]= 1
  return patches_img,patches_topleft[::-1]*[patch_w,patch_h],patch_w,patch_h
//-------------------------------------------------------------------------------------------
*/


// from binary_seg2 import FindSegments
cv::Mat DrawClusters(const cv::Mat &img, const std::list<TClusterNode> &clusters)
{
  cv::Mat img_viz;
  img.convertTo(img_viz, CV_8U);
  cv::cvtColor(img_viz, img_viz, CV_GRAY2BGR);

  cv::Scalar col_set[]= {cv::Scalar(255,0,0),cv::Scalar(0,255,0),cv::Scalar(0,0,255),cv::Scalar(255,255,0),cv::Scalar(0,255,255),cv::Scalar(255,0,255)};
  std::cout<<"clusters:"<<std::endl;
  int i(0);
  for(std::list<TClusterNode>::const_iterator inode(clusters.begin()),inode_end(clusters.end()); inode!=inode_end; ++inode,++i)
  {
    std::cout<<"  "<<i<<" "<<inode->feat<<" patches:"<<inode->patches.size()<<std::endl;
    cv::Scalar col= col_set[i%(sizeof(col_set)/sizeof(col_set[0]))];
    for(std::vector<cv::Rect>::const_iterator ipatch(inode->patches.begin()),ipatch_end(inode->patches.end()); ipatch!=ipatch_end; ++ipatch)
      cv::rectangle(img_viz, *ipatch, col, 1);  //TODO: Shrink the patch for 1px.
    /*
    patches_img,patches_topleft,patch_w,patch_h= PatchPointsToImg(inode->patches)
    #print '  debug:',patches_img.shape
    #if patches_img.size>100:
      #cv2.imwrite('patches_img-{0}.png'.format(i), patches_img)
      #cv2.imshow('patches_img-{0}'.format(i),cv2.resize(patches_img,(patches_img.shape[1]*10,patches_img.shape[0]*10),interpolation=cv2.INTER_NEAREST ))
    segments,num_segments= FindSegments(patches_img)
    print 'seg:',[np.sum(segments==idx) for idx in range(1,num_segments+1)]
    for idx in range(1,num_segments+1):
      patches= [patches_topleft + [u*patch_w,v*patch_h] for v,u in zip(*np.where(segments==idx))]
      for patch in patches:
        x,y= patch
        for j in range(0,idx):
          cv2.circle(img_viz, (x+patch_w/2,y+patch_h/2), max(1,(patch_w+patch_h)/4-2*j), col, 1)

    */
  }
  return img_viz;
}
//-------------------------------------------------------------------------------------------



#define LIBRARY
#include "ros_capture.cpp"

void CVCallback(const cv::Mat &img_depth)
{
  double t_start= GetCurrentTime();
  //std::list<TClusterNode> clusters= ClusteringByFeatures(img_depth, w_patch=25, f_feat=TImgPatchFeatNormal(0.4), th_feat=0.2);
  std::list<TClusterNode> clusters= ClusteringByFeatures(img_depth, /*w_patch=*/25, /*f_feat=*/TImgPatchFeatNormal(5.0), /*th_feat=*/0.5);
  //std::list<TClusterNode> clusters= ClusteringByFeatures(img_depth, w_patch=25, f_feat=TImgPatchFeatAvrDepth(), th_feat=3.0);
  // clusters= [node for node in clusters if len(node.patches)>=3]
  std::cout<<"Number of clusters: "<<clusters.size()<<std::endl;
  //std::cout<<"Sum of numbers of patches: ",sum([len(node.patches) for node in clusters])
  std::cout<<"Computation time: "<<GetCurrentTime()-t_start<<std::endl;
  cv::Mat img_viz= DrawClusters(img_depth, clusters);

  cv::imshow("depth", img_viz);
  char c(cv::waitKey(1));
  if(c=='\x1b'||c=='q')  FinishLoop();
}
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  std::string img_topic("/camera/aligned_depth_to_color/image_raw"), encoding(sensor_msgs::image_encodings::TYPE_16UC1);
  if(argc>1)  img_topic= argv[1];
  if(argc>2)  encoding= argv[2];
  cv::namedWindow("depth",1);
  StartLoop(argc, argv, img_topic, encoding, CVCallback);
  return 0;
}
//-------------------------------------------------------------------------------------------
