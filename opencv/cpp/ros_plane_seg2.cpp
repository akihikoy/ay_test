//-------------------------------------------------------------------------------------------
/*! \file    ros_plane_seg2.cpp
    \brief   Plane segmentation ported from ../python/plane_seg1.py
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Jan.14, 2021

g++ -O2 -g -W -Wall -o ros_plane_seg2.out ros_plane_seg2.cpp  -I../include -I/opt/ros/$ROS_DISTR/include -pthread -llog4cxx -lpthread -L/opt/ros/$ROS_DISTR/lib -rdynamic -lroscpp -lrosconsole -lroscpp_serialization -lrostime -lcv_bridge -lopencv_highgui -lopencv_imgproc -lopencv_core -lopencv_videoio -Wl,-rpath,/opt/ros/$ROS_DISTR/lib

*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <fstream>
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


// Extract effective depth points and store them into Nx3 matrix.
template<typename t_img_depth>
cv::Mat DepthImgToPoints(const cv::Mat &img_patch, const double &d_scale=1.0, int step=1)
{
  // Extract effective depth points.
  int num_data(0);
  for(int r(0);r<img_patch.rows;r+=step)
    for(int c(0);c<img_patch.cols;c+=step)
      if(img_patch.at<t_img_depth>(r,c)>0)  ++num_data;
  cv::Mat points(num_data,3,CV_64F);
  for(int r(0),i(0);r<img_patch.rows;r+=step)
    for(int c(0);c<img_patch.cols;c+=step)
    {
      const double &d= img_patch.at<t_img_depth>(r,c);
      if(d>0)
      {
        points.at<double>(i,0)= c;
        points.at<double>(i,1)= r;
        points.at<double>(i,2)= d * d_scale;
        ++i;
      }
    }
  return points;
}
//-------------------------------------------------------------------------------------------

/*
// Feature definition for clustering (interface class).
class TImgPatchFeatIF
{
public:
  typedef unsigned short TImgDepth;

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
  TImgPatchFeatNormal(const double &th_plane=0.4, const double &d_scale=1.0, int step=1)
    {
      th_plane_= th_plane;
      d_scale_= d_scale;
      step_= step;
    }
  // Get a normal vector of an image patch img_patch.
  cv::Mat Feat(const cv::Mat &img_patch) const
    {
      cv::Mat points= DepthImgToPoints<TImgDepth>(img_patch, d_scale_, step_);
      if(points.rows<3)  return cv::Mat();

      cv::PCA pca(points, cv::Mat(), CV_PCA_DATA_AS_ROW);
      cv::Mat normal= pca.eigenvectors.row(2);
      if(normal.at<double>(0,2)<0)  normal= -normal;
      if(pca.eigenvalues.at<double>(2) > th_plane_ * d_scale_)  return cv::Mat();
std::cerr<<points.rows<<" debug "<<img_patch.type()<<" "<<normal<<" pca.mean: "<<pca.mean<<std::endl;
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
  double d_scale_;
  int step_;
};
//-------------------------------------------------------------------------------------------
*/

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

struct TClusterPatch
{
  cv::Rect ROI;
  bool IsValid;
  cv::Mat Normal;
  cv::Mat Mean;
};
//-------------------------------------------------------------------------------------------

struct TClusterPatchSet
{
  typedef unsigned short TImgDepth;
//   typedef std::pair<int,int> TLoc;  // location.
  int WPatch;
  int Nu, Nv;
//   std::map<TLoc, int> LocToIdx;
  std::vector<TClusterPatch> Patches;

  TClusterPatchSet(void)
    : WPatch(0), Nu(0), Nv(0)  {}

  int UVToIdx(int u, int v) const
    {
      return v*Nu + u;
    }

  void ConstructFromDepthImg(const cv::Mat &img, int w_patch, const double &th_plane=0.4, const double &d_scale=1.0, int step=1)
    {
      if(w_patch!=WPatch || Nu!=img.cols/WPatch || Nv!=img.rows/WPatch || int(Patches.size())!=Nu*Nv)
      {
        WPatch= w_patch;
        Nu= img.cols/WPatch;
        Nv= img.rows/WPatch;
        Patches.resize(Nu*Nv);
//         LocToIdx.clear();
        std::vector<TClusterPatch>::iterator p_itr(Patches.begin());
        for(int v(0)/*,i(0)*/;v<Nv;++v)
          for(int u(0);u<Nu;++u,++p_itr/*,++i*/)
          {
//             LocToIdx[TLoc(u,v)]= i;
            p_itr->ROI= cv::Rect(u*WPatch,v*WPatch,WPatch,WPatch);
          }
      }
      Update(img, th_plane, d_scale, step);
    }
  void Update(const cv::Mat &img, const double &th_plane=0.4, const double &d_scale=1.0, int step=1)
    {
      std::vector<TClusterPatch>::iterator p_itr(Patches.begin());
      for(int v(0);v<Nv;++v)
        for(int u(0);u<Nu;++u,++p_itr)
        {
          TClusterPatch &p(*p_itr);
          p.IsValid= false;

          cv::Mat points= DepthImgToPoints<TImgDepth>(img(p.ROI), d_scale, step);
          if(points.rows<3)  continue;

          cv::PCA pca(points, cv::Mat(), CV_PCA_DATA_AS_ROW);
          p.Normal= pca.eigenvectors.row(2);
          p.Mean= pca.mean;
          p.Mean.at<double>(0,0)+= p.ROI.x;
          p.Mean.at<double>(0,1)+= p.ROI.y;
          if(p.Normal.at<double>(0,2)<0)  p.Normal= -p.Normal;
          if(pca.eigenvalues.at<double>(2) > th_plane * d_scale)  continue;
          p.IsValid= true;
        }
    }
};
//-------------------------------------------------------------------------------------------


// Feature definition for clustering (interface class).
class TClusteringFeatIF
{
public:
  // Set up the feature.  Parameters may be added.
  TClusteringFeatIF()  {}
  // Get a feature vector for a patch.  Return cv::Mat() for an invalid patch.
  virtual cv::Mat Feat(const TClusterPatch &patch) const = 0;
  // Get a difference (scholar value) between two features.
  virtual double Diff(const cv::Mat &f1, const cv::Mat &f2) const = 0;
  // Compute a weighted sum of two features. i.e. w1*f1 + (1.0-w1)*f2
  virtual cv::Mat WSum(const cv::Mat &f1, const cv::Mat &f2, const double &w1) const = 0;
};
//-------------------------------------------------------------------------------------------

//Feature of the normal of a patch.
class TClusteringFeatNormal : public TClusteringFeatIF
{
public:
  TClusteringFeatNormal()
    : TClusteringFeatIF()
    {
    }
  // Get a feature vector for a patch.  Return cv::Mat() for an invalid patch.
  virtual cv::Mat Feat(const TClusterPatch &patch) const
    {
      if(!patch.IsValid)  return cv::Mat();
      return patch.Normal;
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
        CV_Error(CV_StsError, "TClusteringFeatNormal: Computing WSum for normals of opposite directions.");
      return ws/ws_norm;
    }
private:
};
//-------------------------------------------------------------------------------------------

struct TClusterNode
{
  std::vector<int> Patches;
  cv::Mat Feat;
  std::set<int> Neighbors;  // Neighbor nodes that are candidate to be merged.

  TClusterNode()  {}
  TClusterNode(const int &patch0, const cv::Mat feat0)
    {
      Patches.push_back(patch0);
      Feat= feat0;
    }
};
//-------------------------------------------------------------------------------------------

// Segment img by a given feature model such as normal.
void ClusteringByFeatures(const TClusterPatchSet &patch_set, std::vector<TClusterNode> &clusters,
  /*const cv::Mat &img,*/ const TClusteringFeatIF &f_feat, const double &th_feat=15.0)
{
  // img= img.reshape((img.shape[0],img.shape[1]))

  /*
  depth image --> patches (w_patch x w_patch); each patch has a feat.
  patches --> nodes = initial planes
  nodes= [TClusterNode([(x,y,w_patch,w_patch)],f_feat(img[y:y+w_patch,x:x+w_patch]))
          for y in range(0,img.shape[0],w_patch)
          for x in range(0,img.shape[1],w_patch)]
  */
//   int Nu= img.cols/w_patch;
//   int Nv= img.rows/w_patch;
  std::vector<TClusterNode> nodes(patch_set.Patches.size());
  for(int i(0),i_end(patch_set.Patches.size()); i<i_end; ++i)
    nodes[i]= TClusterNode(i,f_feat.Feat(patch_set.Patches[i]));

//   typedef std::pair<int,int> TLoc;  // location.
  typedef int TNodeItr;
//   std::map<TLoc, TNodeItr> node_map;
//   for(int v(0);v<Nv;++v)
//     for(int u(0);u<Nu;++u)
//     {
//       cv::Rect patch(u*w_patch,v*w_patch,w_patch,w_patch);
//       cv::Mat feat= f_feat.Feat(img(patch));
//       nodes.push_back(TClusterNode(patch,feat));
//       TNodeItr itr= nodes.end();
//       --itr;
//       node_map[TLoc(u,v)]= nodes.size()-1;
//     }
//   /*DEBUG*/std::cerr<<"DEBUG:Nu,Nv: "<<Nu<<", "<<Nv<<std::endl;
  // int dneighbors[][2]= ((1,1),(1,0),(1,-1),(0,1),(0,-1),(-1,1),(-1,0),(-1,-1))
  int dneighbors[][2]= {{1,0},{0,1},{0,-1},{-1,0}};
  for(int v(0);v<patch_set.Nv;++v)
    for(int u(0);u<patch_set.Nu;++u)
    {
      TNodeItr inode= patch_set.UVToIdx(u,v);
      if(nodes[inode].Feat.empty())  continue;
      for(int inei(0);inei<4;++inei)
      {
        int du(dneighbors[inei][0]), dv(dneighbors[inei][1]);
        if(0<=u+du && u+du<patch_set.Nu && 0<=v+dv && v+dv<patch_set.Nv
                && !nodes[patch_set.UVToIdx(u+du,v+dv)].Feat.empty())
          nodes[inode].Neighbors.insert(patch_set.UVToIdx(u+du,v+dv));
      }
    }
  // nodes= filter(lambda node:node.Feat is not None, nodes)
//   for(TNodeItr itr(nodes.end()); itr!=nodes.begin();)
//   {
//     --itr;
//     if(itr->Feat.empty())  itr= a.erase(itr);
//   }

  // Clustering nodes.
  std::list<TNodeItr> inodes, iclusters;
  for(TNodeItr itr(0),itr_end(nodes.size()); itr!=itr_end; ++itr)
    if(!nodes[itr].Feat.empty())  inodes.push_back(itr);
  while(!inodes.empty())
  {
    TNodeItr inode= inodes.front();  inodes.pop_front();  //FIXME: pop a random element?
    std::set<TNodeItr> neighbors= nodes[inode].Neighbors;
    nodes[inode].Neighbors= std::set<TNodeItr>();
    for(std::set<TNodeItr>::iterator iinode2(neighbors.begin()),iinode2_end(neighbors.end()); iinode2!=iinode2_end; ++iinode2)
    {
      TNodeItr inode2(*iinode2);
      nodes[inode2].Neighbors.erase(inode);
// std::cerr<<"debug:f_feat.Diff: "<<f_feat.Diff(nodes[inode].Feat,nodes[inode2].Feat)<<std::endl;
      if(f_feat.Diff(nodes[inode].Feat,nodes[inode2].Feat) < th_feat)
      {
        // Merge inode2 into inode:
        double r1= double(nodes[inode].Patches.size())/double(nodes[inode].Patches.size()+nodes[inode2].Patches.size());
        nodes[inode].Feat= f_feat.WSum(nodes[inode].Feat, nodes[inode2].Feat, r1);
        nodes[inode].Patches.insert(nodes[inode].Patches.end(), nodes[inode2].Patches.begin(), nodes[inode2].Patches.end());
        // nodes[inode].Neighbors.update(nodes[inode2].Neighbors-neighbors)
        std::set<TNodeItr> n_diff;
        std::set_difference(nodes[inode2].Neighbors.begin(),nodes[inode2].Neighbors.end(), neighbors.begin(),neighbors.end(), std::inserter(n_diff,n_diff.begin()));
        nodes[inode].Neighbors.insert(n_diff.begin(),n_diff.end());
        if(std::find(inodes.begin(),inodes.end(),inode2)!=inodes.end())  inodes.remove(inode2);
        for(std::set<TNodeItr>::iterator iinode3(nodes[inode2].Neighbors.begin()),iinode3_end(nodes[inode2].Neighbors.end()); iinode3!=iinode3_end; ++iinode3)
          nodes[*iinode3].Neighbors.erase(inode2);
        n_diff.clear();
        std::set_difference(nodes[inode2].Neighbors.begin(),nodes[inode2].Neighbors.end(), neighbors.begin(),neighbors.end(), std::inserter(n_diff,n_diff.begin()));
        for(std::set<TNodeItr>::iterator iinode3(n_diff.begin()),iinode3_end(n_diff.end()); iinode3!=iinode3_end; ++iinode3)
          nodes[*iinode3].Neighbors.insert(inode);
      }
    }
// std::cerr<<"debug:inode: "<<inode<<" "<<*nodes[inode].Neighbors.begin()<<std::endl;
    if(!nodes[inode].Neighbors.empty())
      inodes.push_back(inode);
    else
      iclusters.push_back(inode);
  }
  clusters.clear();
  clusters.reserve(iclusters.size());
  for(std::list<TNodeItr>::iterator itr(iclusters.begin()),itr_end(iclusters.end()); itr!=itr_end; ++itr)
    clusters.push_back(nodes[*itr]);
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


// Get a plane model from depth image and patches.
void GetPlaneFromPatches(const TClusterPatchSet &patch_set, const std::vector<int> &patches,
  cv::Mat &out_normal, cv::Mat &out_center)
{
  if(patches.size()<3)  return;
  cv::Mat points(patches.size(),3,CV_64F);
  int i(0);
  for(std::vector<int>::const_iterator ip(patches.begin()),ip_end(patches.end()); ip!=ip_end; ++ip,++i)
  {
    points.at<double>(i,0)= patch_set.Patches[*ip].Mean.at<double>(0,0);
    points.at<double>(i,1)= patch_set.Patches[*ip].Mean.at<double>(0,1);
    points.at<double>(i,2)= patch_set.Patches[*ip].Mean.at<double>(0,2);
  }

  /*DEBUG:Save points:*-/
  std::string filename("/tmp/points.dat");
  {
    std::ofstream ofs(filename.c_str());
    std::string delim;
    for(int r(0);r<points.rows;++r)
    {
      delim= "";
      for(int c(0);c<points.cols;++c)
      {
        ofs<<delim<<points.at<double>(r,c);
        delim= " ";
      }
      ofs<<std::endl;
    }
  }//*/

  cv::PCA pca(points, cv::Mat(), CV_PCA_DATA_AS_ROW);
  out_normal= pca.eigenvectors.row(2);
  out_center= pca.mean;
  if(out_normal.at<double>(0,2)<0)  out_normal= -out_normal;
}
//-------------------------------------------------------------------------------------------

// Extract depth around a plane whose depth at (u,v) is given by D(normal,center,u,v).
// For each pixel (u,v), points [D(normal,center,u,v)+lower, D(normal,center,u,v)+upper] are extracted.
void DepthExtractAroundPlane(cv::Mat &img_depth, const cv::Mat &normal, const cv::Mat &center, const double &lower, const double &upper)
{
  double u0(center.at<double>(0,0)), v0(center.at<double>(0,1)), d0(center.at<double>(0,2));
  double N0(normal.at<double>(0,0)), N1(normal.at<double>(0,1)), N2(normal.at<double>(0,2));
// std::cerr<<"debug:img_depth.size:"<<img_depth.type()<<" "<<img_depth.rows<<" "<<img_depth.cols<<std::endl;
// std::cerr<<"debug:center:"<<u0<<" "<<v0<<" "<<d0<<" N:"<<N0<<" "<<N1<<" "<<N2<<std::endl;
  for(int v(0);v<img_depth.rows;++v)
    for(int u(0);u<img_depth.cols;++u)
    {
// std::cerr<<"debug:u,v: "<<u<<" "<<v<<std::endl;
// std::cerr<<"debug:d: "<<img_depth.at<unsigned short>(v,u)<<std::endl;
// std::cerr<<"--"<<std::endl;
      double d= d0 - (N0*(u-u0)+N1*(v-v0))/N2;
// std::cerr<<" "<<d;
      if(img_depth.at<unsigned short>(v,u) < d+lower || d+upper < img_depth.at<unsigned short>(v,u))
        img_depth.at<unsigned short>(v,u)= 0.0;
    }
// std::cerr<<std::endl;
}
//-------------------------------------------------------------------------------------------

// from binary_seg2 import FindSegments
cv::Mat DrawClusters(const cv::Mat &img, const TClusterPatchSet &patch_set, const std::vector<TClusterNode> &clusters, bool disp_info)
{
  cv::Mat img_viz(img*0.3);
  img_viz.convertTo(img_viz, CV_8U);
  cv::cvtColor(img_viz, img_viz, CV_GRAY2BGR);

  cv::Scalar col_set[]= {cv::Scalar(255,0,0),cv::Scalar(0,255,0),cv::Scalar(0,0,255),cv::Scalar(255,255,0),cv::Scalar(0,255,255),cv::Scalar(255,0,255)};
  if(disp_info)  std::cout<<"clusters:"<<std::endl;
  int ic_largest(-1);
  for(int ic(0),ic_end(clusters.size()); ic<ic_end; ++ic)
  {
    if(clusters[ic].Patches.size()<5)  continue;
    if(ic_largest<0 || clusters[ic_largest].Patches.size()<clusters[ic].Patches.size())
      ic_largest= ic;
    if(disp_info)  std::cout<<"  "<<ic<<" "<<clusters[ic].Neighbors.size()<<" feat:"<<clusters[ic].Feat<<" patches:"<<clusters[ic].Patches.size()<<std::endl;
    cv::Scalar col= col_set[ic%(sizeof(col_set)/sizeof(col_set[0]))];
    for(int ip(0),ip_end(clusters[ic].Patches.size());ip<ip_end;++ip)
      cv::rectangle(img_viz, patch_set.Patches[clusters[ic].Patches[ip]].ROI, col, 1);  //TODO: Shrink the patch for 1px.
    /*
    patches_img,patches_topleft,patch_w,patch_h= PatchPointsToImg(clusters[ic].Patches)
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
  if(ic_largest>=0)
  {
    // Save points of a patch in the largest cluster:
    // cv::Mat img_patch= img(patch_set.Patches[clusters[ic_largest].Patches[clusters[ic_largest].Patches.size()/2]].ROI);
    // cv::Mat data= DepthImgToPoints<unsigned short>(img_patch, /*d_scale*/1.0, /*step*/1);
// std::cerr<<"debug:ic_largest:patch:"<<patch_set.Patches[clusters[ic_largest].Patches[clusters[ic_largest].Patches.size()/2]].ROI<<std::endl;
// std::cerr<<"img_patch:"<<img_patch<<std::endl;
// std::cerr<<"data:"<<data<<std::endl;
    // std::string filename("/tmp/points.dat");
    // {
    //   std::ofstream ofs(filename.c_str());
    //   std::string delim;
    //   for(int r(0);r<data.rows;++r)
    //   {
    //     delim= "";
    //     for(int c(0);c<data.cols;++c)
    //     {
    //       ofs<<delim<<data.at<double>(r,c);
    //       delim= " ";
    //     }
    //     ofs<<std::endl;
    //   }
    // }
  }
  return img_viz;
}
//-------------------------------------------------------------------------------------------



#define LIBRARY
#include "ros_capture.cpp"
#include "float_trackbar.cpp"

namespace ns_main
{
TClusterPatchSet patch_set;
bool disp_info(true);
int w_patch(25);
double th_plane(0.4);
double th_feat(0.2);
// double lower(-10.0), upper(80.0);  // Extract plane.
double lower(-300.0), upper(-24.0);  // Extract objects on plane.

void Init(int /*argc*/, char**/*argv*/)
{
  cv::namedWindow("depth",1);

  CreateTrackbar<bool>("disp_info", "depth", &disp_info, &TrackbarPrintOnTrack);
  CreateTrackbar<int>("w_patch", "depth", &w_patch, 1, 51, 2,  &TrackbarPrintOnTrack);
  CreateTrackbar<double>("th_plane", "depth", &th_plane, 0.0, 5.0, 0.01,  &TrackbarPrintOnTrack);
  CreateTrackbar<double>("th_feat", "depth", &th_feat, 0.0, 5.0, 0.01,  &TrackbarPrintOnTrack);
  CreateTrackbar<double>("lower", "depth", &lower, -500.0, 500.0, 0.1,  &TrackbarPrintOnTrack);
  CreateTrackbar<double>("upper", "depth", &upper, -500.0, 500.0, 0.1,  &TrackbarPrintOnTrack);
}
//-------------------------------------------------------------------------------------------

void CVCallback(const cv::Mat &img_depth)
{
  double t_start= GetCurrentTime();
  patch_set.ConstructFromDepthImg(img_depth, w_patch, th_plane);
  if(disp_info)  std::cerr<<"Patch set cmp time: "<<GetCurrentTime()-t_start<<std::endl;
  std::vector<TClusterNode> clusters;
  ClusteringByFeatures(patch_set, clusters, TClusteringFeatNormal(), th_feat);
  // clusters= [node for node in clusters if len(node.Patches)>=3]
  if(disp_info)  std::cout<<"Number of clusters: "<<clusters.size()<<std::endl;
  //std::cout<<"Sum of numbers of patches: ",sum([len(node.Patches) for node in clusters])
  if(disp_info)  std::cout<<"Computation time: "<<GetCurrentTime()-t_start<<std::endl;
  cv::Mat img_viz= DrawClusters(img_depth, patch_set, clusters, disp_info);

  // Extract points around the largest plane:
  int ic_largest(-1);
  for(int ic(0),ic_end(clusters.size()); ic<ic_end; ++ic)
  {
    if(clusters[ic].Patches.size()<5)  continue;
    if(ic_largest<0 || clusters[ic_largest].Patches.size()<clusters[ic].Patches.size())
      ic_largest= ic;
  }
  if(ic_largest>=0)
  {
    cv::Mat normal, center, img_depth2;
    GetPlaneFromPatches(patch_set, clusters[ic_largest].Patches, normal, center);
    if(disp_info)  std::cout<<"Plane: "<<normal<<", "<<center<<std::endl;
    img_depth.copyTo(img_depth2);
    DepthExtractAroundPlane(img_depth2, normal, center, lower, upper);
    cv::imshow("depth_extracted", img_depth2*255.0*0.3);
  }

  cv::imshow("depth", img_viz);
  char c(cv::waitKey(1));
  if(c=='\x1b'||c=='q')  FinishLoop();
}
//-------------------------------------------------------------------------------------------

} //namespace ns_main
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  std::string img_topic("/camera/aligned_depth_to_color/image_raw"), encoding(sensor_msgs::image_encodings::TYPE_16UC1);
  if(argc>1)  img_topic= argv[1];
  if(argc>2)  encoding= argv[2];
  ns_main::Init(argc, argv);
  StartLoop(argc, argv, img_topic, encoding, ns_main::CVCallback);
  return 0;
}
//-------------------------------------------------------------------------------------------
