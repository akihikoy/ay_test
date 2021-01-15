//-------------------------------------------------------------------------------------------
/*! \file    ros_plane_seg3.cpp
    \brief   Plane segmentation ported from ../python/plane_seg1.py
             The normal vectors are computed from 3D points by transforming with camera matrix.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Jan.14, 2021

g++ -O2 -g -W -Wall -o ros_plane_seg3.out ros_plane_seg3.cpp  -I../include -I/opt/ros/kinetic/include -pthread -llog4cxx -lpthread -L/opt/ros/kinetic/lib -rdynamic -lroscpp -lrosconsole -lroscpp_serialization -lrostime -lcv_bridge -lopencv_highgui -lopencv_imgproc -lopencv_core -Wl,-rpath,/opt/ros/kinetic/lib

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
#include <cmath>
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


// Extract effective depth points and store them into Nx3 matrix.
template<typename t_img_depth>
cv::Mat DepthImgTo3DPoints(const cv::Rect &roi, const cv::Mat &img_patch, const cv::Mat &proj_mat, int step=1)
{
  double Fx,Fy,Cx,Cy;
  Fx= proj_mat.at<double>(0,0);
  Fy= proj_mat.at<double>(1,1);
  Cx= proj_mat.at<double>(0,2);
  Cy= proj_mat.at<double>(1,2);

  // Extract effective depth points.
  int num_data(0);
  for(int r(0);r<img_patch.rows;r+=step)
    for(int c(0);c<img_patch.cols;c+=step)
      if(img_patch.at<t_img_depth>(r,c)>0)  ++num_data;
  cv::Mat points(num_data,3,CV_64F);
  for(int r(0),i(0);r<img_patch.rows;r+=step)
    for(int c(0);c<img_patch.cols;c+=step)
    {
      const double d= img_patch.at<t_img_depth>(r,c) * 0.001;
      if(d>0)
      {
        points.at<double>(i,0)= (roi.x+c-Cx)/Fx*d;
        points.at<double>(i,1)= (roi.y+r-Cy)/Fy*d;
        points.at<double>(i,2)= d;
        ++i;
      }
    }
  return points;
}
//-------------------------------------------------------------------------------------------


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
  int WPatch;
  int Nu, Nv;
  std::vector<TClusterPatch> Patches;

  TClusterPatchSet(void)
    : WPatch(0), Nu(0), Nv(0)  {}

  int UVToIdx(int u, int v) const
    {
      return v*Nu + u;
    }

  void ConstructFromDepthImg(const cv::Mat &img, const cv::Mat &proj_mat, int w_patch, const double &th_plane=0.01, int step=1)
    {
      if(w_patch!=WPatch || Nu!=img.cols/WPatch || Nv!=img.rows/WPatch || int(Patches.size())!=Nu*Nv)
      {
        WPatch= w_patch;
        Nu= img.cols/WPatch;
        Nv= img.rows/WPatch;
        Patches.resize(Nu*Nv);
        std::vector<TClusterPatch>::iterator p_itr(Patches.begin());
        for(int v(0);v<Nv;++v)
          for(int u(0);u<Nu;++u,++p_itr)
          {
            p_itr->ROI= cv::Rect(u*WPatch,v*WPatch,WPatch,WPatch);
          }
      }
      Update(img, proj_mat, th_plane, step);
    }
  void Update(const cv::Mat &img, const cv::Mat &proj_mat, const double &th_plane=0.01, int step=1)
    {
      std::vector<TClusterPatch>::iterator p_itr(Patches.begin());
      for(int v(0);v<Nv;++v)
        for(int u(0);u<Nu;++u,++p_itr)
        {
          TClusterPatch &p(*p_itr);
          p.IsValid= false;

          cv::Mat points= DepthImgTo3DPoints<TImgDepth>(p.ROI, img(p.ROI), proj_mat, step);
          if(points.rows<3)  continue;

          cv::PCA pca(points, cv::Mat(), CV_PCA_DATA_AS_ROW);
          p.Normal= pca.eigenvectors.row(2);
          p.Mean= pca.mean;
          if(p.Normal.at<double>(0,2)<0)  p.Normal= -p.Normal;
          if(pca.eigenvalues.at<double>(2) > th_plane)  continue;
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
  std::vector<TClusterNode> nodes(patch_set.Patches.size());
  for(int i(0),i_end(patch_set.Patches.size()); i<i_end; ++i)
    nodes[i]= TClusterNode(i,f_feat.Feat(patch_set.Patches[i]));

  typedef int TNodeItr;
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

  /*DEBUG:Save points:*/
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

// Extract points whose height from the plane given by normal and center is within [lower,upper].
void DepthExtractAroundPlane(cv::Mat &img_depth, const cv::Mat &proj_mat, const cv::Mat &normal, const cv::Mat &center, const double &lower, const double &upper)
{
  // Plane parameters:
  double x0(center.at<double>(0,0)), y0(center.at<double>(0,1)), z0(center.at<double>(0,2));
  double N0(normal.at<double>(0,0)), N1(normal.at<double>(0,1)), N2(normal.at<double>(0,2));
  // Camera matrix:
  double Fx,Fy,Cx,Cy;
  Fx= proj_mat.at<double>(0,0);
  Fy= proj_mat.at<double>(1,1);
  Cx= proj_mat.at<double>(0,2);
  Cy= proj_mat.at<double>(1,2);
  for(int v(0);v<img_depth.rows;++v)
    for(int u(0);u<img_depth.cols;++u)
    {
      // u,v,d --> 3D
      const double pz= img_depth.at<unsigned short>(v,u) * 0.001;
      const double px= (u-Cx)/Fx*pz;
      const double py= (v-Cy)/Fy*pz;
      // height from the plane:
      const double h= N0*(px-x0) + N1*(py-y0) + N2*(pz-z0);
      if(h < lower || upper < h)
        img_depth.at<unsigned short>(v,u)= 0.0;
    }
}
//-------------------------------------------------------------------------------------------

void MakeMaskFromContour(const std::vector<std::vector<cv::Point> > &contours, int ic, cv::Mat &mask, int fill_value=1)
{
  if(ic<0 || ic>=int(contours.size()))  return;
//   double a(0.0),a_max(0.0), i_max(0);
//   for(int i(0),i_end(contours.size()); i<i_end; ++i)
//   {
//     a= cv::contourArea(contours[i],false);
//     if(a>a_max)  {a_max= a;  i_max= i;}
//   }
  cv::drawContours(mask, contours, ic, fill_value, /*thickness=*/-1);
//   if(!convex)
//     cv::drawContours(mask, contours, ic, fill_value, /*thickness=*/-1);
//   else
//   {
//     std::vector<std::vector<cv::Point> > hull(1);
//     cv::convexHull(contours[ic], hull[0], /*clockwise=*/true);
//     cv::drawContours(mask, hull, 0, fill_value, /*thickness=*/-1);
//   }
}
//-------------------------------------------------------------------------------------------

// Fit a cylinder parameter to an object specified by img_depth(mask,roi) on a plane specified by normal_pl and center_pl.
// NOTE: The direction cylinder==normal_pl.
void FitCylinder(const cv::Mat &img_depth, const cv::Mat &proj_mat, const cv::Mat &mask, const cv::Rect &roi, const cv::Mat &normal_pl, const cv::Mat &center_pl, double &h_cyl, double &r_cyl, cv::Mat &center_cyl)
{
  // Extract object depth to be fitted.
  cv::Mat img_depth_obj(roi.size(), img_depth.type());
  img_depth_obj.setTo(0);
  img_depth(roi).copyTo(img_depth_obj, mask(roi));

// cv::imshow("tmp", img_depth_obj*255.0*0.3);

  // Plane parameters:
  double x0(center_pl.at<double>(0,0)), y0(center_pl.at<double>(0,1)), z0(center_pl.at<double>(0,2));
  double N0(normal_pl.at<double>(0,0)), N1(normal_pl.at<double>(0,1)), N2(normal_pl.at<double>(0,2));
  // Camera matrix:
  double Fx,Fy,Cx,Cy;
  Fx= proj_mat.at<double>(0,0);
  Fy= proj_mat.at<double>(1,1);
  Cx= proj_mat.at<double>(0,2);
  Cy= proj_mat.at<double>(1,2);

  h_cyl= 0.0;
  center_cyl.create(center_pl.size(), center_pl.type());
  center_cyl.setTo(0.0);
  std::list<cv::Point3d> points_pl;
  for(int v(0);v<img_depth_obj.rows;++v)
    for(int u(0);u<img_depth_obj.cols;++u)
    {
      if(img_depth_obj.at<unsigned short>(v,u)<=0)  continue;
      // u,v,d --> 3D
      const double pz= img_depth_obj.at<unsigned short>(v,u) * 0.001;
      const double px= (roi.x+u-Cx)/Fx*pz;
      const double py= (roi.y+v-Cy)/Fy*pz;
      // height from the plane:
      const double h_pl= N0*(px-x0) + N1*(py-y0) + N2*(pz-z0);
      if(std::fabs(h_pl)>h_cyl)  h_cyl= std::fabs(h_pl);
      // Point projected onto the plane:
      const double px_pl= px - h_pl*N0;
      const double py_pl= py - h_pl*N1;
      const double pz_pl= pz - h_pl*N2;
      center_cyl.at<double>(0,0)+= px_pl;
      center_cyl.at<double>(0,1)+= py_pl;
      center_cyl.at<double>(0,2)+= pz_pl;
      points_pl.push_back(cv::Point3d(px_pl,py_pl,pz_pl));
    }
  center_cyl/= static_cast<double>(points_pl.size());
  r_cyl= 0.0;
  double r(0.0);
  for(std::list<cv::Point3d>::const_iterator ip_pl(points_pl.begin()),ip_pl_end(points_pl.end()); ip_pl!=ip_pl_end; ++ip_pl)
  {
    r= cv::norm(*ip_pl - cv::Point3d(center_cyl));
    if(r>r_cyl)  r_cyl= r;
  }
}
//-------------------------------------------------------------------------------------------

void DepthSegmentation(const cv::Mat &img_depth, std::vector<std::vector<cv::Point> > &contours, int n_dilate=10, int n_erode=5, bool convex=false)
{
  cv::Mat img_depth2;
  img_depth.convertTo(img_depth2, CV_8U);
  if(n_dilate>0)  cv::dilate(img_depth2,img_depth2,cv::Mat(),cv::Point(-1,-1), n_dilate);
  if(n_erode>0)   cv::erode(img_depth2,img_depth2,cv::Mat(),cv::Point(-1,-1), n_erode);

  // Contour detection
  contours.clear();
  cv::findContours(img_depth2, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

  if(convex && contours.size()>0)
  {
    std::vector<std::vector<cv::Point> > hull(1);
    for(int ic(0),ic_end(contours.size()); ic<ic_end; ++ic)
    {
      hull[0].clear();
      cv::convexHull(contours[ic], hull[0], /*clockwise=*/true);
      cv::drawContours(img_depth2, hull, 0, /*fill_value=*/255, /*thickness=*/-1);
    }
    contours.clear();
    cv::findContours(img_depth2, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
  }

  if(contours.size()>0)
  {
    for(int ic(0),ic_end(contours.size()); ic<ic_end; ++ic)
    {
      // double area= cv::contourArea(contours[ic],false);
      cv::Rect roi= cv::boundingRect(contours[ic]);
//       int bound_len= std::max(roi.width, roi.height);
//       if(bound_len<rect_len_min || bound_len>rect_len_max)  continue;
      cv::drawContours(img_depth2, contours, ic, CV_RGB(255,0,255), /*thickness=*/2, /*linetype=*/8);
      cv::rectangle(img_depth2, roi, cv::Scalar(0,0,255), 2);
    }
  }
cv::imshow("test_segmentation", img_depth2*255.0*0.3);
}
//-------------------------------------------------------------------------------------------

void DrawCylinder(cv::Mat &img_depth, const cv::Mat &proj_mat, const double &h, const double &r, const cv::Mat &center, const cv::Mat &dir)
{
  // Camera matrix:
  double Fx,Fy,Cx,Cy;
  Fx= proj_mat.at<double>(0,0);
  Fy= proj_mat.at<double>(1,1);
  Cx= proj_mat.at<double>(0,2);
  Cy= proj_mat.at<double>(1,2);

  // std::vector<std::vector<cv::Point> >  points1(1), points2(1);
  // for(double theta(0.0); theta<2.0*MATH_PI; ++theta)
  // {
  // }
  // // cv::fillPoly(img_depth, points1, CV_RGB(128,0,128));
  // cv::polylines(img_depth, points1, /*isClosed=*/true, CV_RGB(255,0,255), 2);
  // cv::polylines(img_depth, points2, /*isClosed=*/true, CV_RGB(255,0,255), 2);

  cv::Point pb(Fx*center.at<double>(0,0)/center.at<double>(0,2) + Cx,
               Fy*center.at<double>(0,1)/center.at<double>(0,2) + Cy);
  cv::circle(img_depth, pb, 3, CV_RGB(255,0,255), 1);

  cv::Mat top= center+h*dir;
  cv::Point pt(Fx*top.at<double>(0,0)/top.at<double>(0,2) + Cx,
               Fy*top.at<double>(0,1)/top.at<double>(0,2) + Cy);
  cv::circle(img_depth, pb, 3, CV_RGB(255,0,255), 1);
}
//-------------------------------------------------------------------------------------------

void DepthFitCylinders(const cv::Mat &img_depth, const cv::Mat &proj_mat, const std::vector<std::vector<cv::Point> > &contours, const cv::Mat &normal_pl, const cv::Mat &center_pl)
{
  cv::Mat img_depth2(img_depth*255.0*0.3);
  img_depth2.convertTo(img_depth2, CV_8U);
  cv::cvtColor(img_depth2, img_depth2, CV_GRAY2BGR);

  if(contours.size()>0)
  {
    for(int ic(0),ic_end(contours.size()); ic<ic_end; ++ic)
    {
      cv::Mat mask(img_depth.size(), CV_8UC1);
      mask.setTo(0);
      MakeMaskFromContour(contours, ic, mask);
      cv::Rect roi= cv::boundingRect(contours[ic]);
      double h_cyl(0.0), r_cyl(0.0);
      cv::Mat center_cyl;
      FitCylinder(img_depth, proj_mat, mask, roi, normal_pl, center_pl, h_cyl, r_cyl, center_cyl);
DrawCylinder(img_depth2, proj_mat, h_cyl, r_cyl, center_cyl, normal_pl);
    }
  }
cv::imshow("test_cylinders", img_depth2);
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
#include "ros_proj_mat.cpp"

namespace ns_main
{
TClusterPatchSet patch_set;
std::string frame_id;
cv::Mat proj_mat;

bool disp_info(false);
int w_patch(25);
double th_plane(0.01);
double th_feat(0.2);
// double lower(-0.01), upper(0.01);  // Extract plane.
double lower(-0.3), upper(-0.01);  // Extract objects on plane.
int n_dilate(10), n_erode(5);  // Segmentation parameters.
bool convex(false);  // Convert object contour to be convex.

void Init(int argc, char**argv)
{
  std::string cam_info_topic("/camera/aligned_depth_to_color/camera_info");
  if(argc>3)  cam_info_topic= argv[3];

  GetCameraProjectionMatrix(cam_info_topic, frame_id, proj_mat);
  std::cerr<<"frame_id: "<<frame_id<<std::endl;
  std::cerr<<"proj_mat: "<<proj_mat<<std::endl;

  cv::namedWindow("depth",1);

  CreateTrackbar<bool>("disp_info", "depth", &disp_info, &TrackbarPrintOnTrack);
  CreateTrackbar<int>("w_patch", "depth", &w_patch, 1, 51, 2,  &TrackbarPrintOnTrack);
  CreateTrackbar<double>("th_plane", "depth", &th_plane, 0.0, 1.0, 0.001,  &TrackbarPrintOnTrack);
  CreateTrackbar<double>("th_feat", "depth", &th_feat, 0.0, 5.0, 0.01,  &TrackbarPrintOnTrack);
  CreateTrackbar<double>("lower", "depth", &lower, -1.0, 1.0, 0.001,  &TrackbarPrintOnTrack);
  CreateTrackbar<double>("upper", "depth", &upper, -1.0, 1.0, 0.001,  &TrackbarPrintOnTrack);

  CreateTrackbar<int>("n_dilate", "depth", &n_dilate, 0, 50, 1,  &TrackbarPrintOnTrack);
  CreateTrackbar<int>("n_erode", "depth", &n_erode, 0, 50, 1,  &TrackbarPrintOnTrack);
  CreateTrackbar<bool>("convex", "depth", &convex, &TrackbarPrintOnTrack);
}
//-------------------------------------------------------------------------------------------

void CVCallback(const cv::Mat &img_depth)
{
  double t_start= GetCurrentTime();
  patch_set.ConstructFromDepthImg(img_depth, proj_mat, w_patch, th_plane);
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
    cv::Mat normal_pl, center_pl, img_depth2;
    GetPlaneFromPatches(patch_set, clusters[ic_largest].Patches, normal_pl, center_pl);
    if(disp_info)  std::cout<<"Plane: "<<normal_pl<<", "<<center_pl<<std::endl;
    img_depth.copyTo(img_depth2);
    DepthExtractAroundPlane(img_depth2, proj_mat, normal_pl, center_pl, lower, upper);
//     cv::imshow("depth_extracted", img_depth2*255.0*0.3);
    std::vector<std::vector<cv::Point> > contours;
    DepthSegmentation(img_depth2, contours, n_dilate, n_erode, convex);
    DepthFitCylinders(img_depth2, proj_mat, contours, normal_pl, center_pl);
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
  std::string node_name("img_node");
  ros::init(argc, argv, node_name);
  ns_main::Init(argc, argv);
  StartLoop(argc, argv, img_topic, encoding, ns_main::CVCallback, node_name);
  return 0;
}
//-------------------------------------------------------------------------------------------
