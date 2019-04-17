//-------------------------------------------------------------------------------------------
/*! \file    depthscene.cpp
    \brief   Depth scene for ray tracing using Don Cross's library (doncross).
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Feb.03, 2015
*/
//-------------------------------------------------------------------------------------------
#include "depthscene.h"
//-------------------------------------------------------------------------------------------
namespace Imager
{
using namespace std;
// using namespace boost;

//===========================================================================================
// The depth scene object that renders a depth image of objects
// class DepthScene
//===========================================================================================

// Empties out the solidObjectList and destroys/frees
// the SolidObjects that were in it.
void DepthScene::ClearSolidObjectList()
{
    SolidObjectList::iterator iter = solidObjectList.begin();
    SolidObjectList::iterator end  = solidObjectList.end();
    for (; iter != end; ++iter)
    {
        delete *iter;
        *iter = NULL;
    }
    solidObjectList.clear();
}
//-------------------------------------------------------------------------------------------

Intersection DepthScene::TraceRay(
    const Vector& vantage,
    const Vector& direction) const
{
  Intersection intersection;
  const int numClosest = FindClosestIntersection(
      vantage,
      direction,
      intersection);

  switch (numClosest)
  {
  case 0:
    return backgroundIntersection;
  case 1:
    return intersection;
  default:
    // There is an ambiguity: more than one intersection
    // has the same minimum distance.
    // We just use intersection now.
    return intersection;
  }
}
//-------------------------------------------------------------------------------------------

// Searches for an intersections with any solid in the scene from the
// vantage point in the given direction.  If none are found, the
// function returns 0 and the 'intersection' parameter is left
// unchanged.  Otherwise, returns the positive number of
// intersections that lie at minimal distance from the vantage point
// in that direction.  Usually this number will be 1 (a unique
// intersection is closer than all the others) but it can be greater
// if multiple intersections are equally close (e.g. the ray hitting
// exactly at the corner of a cube could cause this function to
// return 3).  If this function returns a value greater than zero,
// it means the 'intersection' parameter has been filled in with the
// closest intersection (or one of the equally closest intersections).
int DepthScene::FindClosestIntersection(
    const Vector& vantage,
    const Vector& direction,
    Intersection& intersection) const
{
    // Build a list of all intersections from all objects.
    cachedIntersectionList.clear();     // empty any previous contents
    SolidObjectList::const_iterator iter = solidObjectList.begin();
    SolidObjectList::const_iterator end  = solidObjectList.end();
    for (; iter != end; ++iter)
    {
        const SolidObject& solid = *(*iter);
        solid.AppendAllIntersections(
            vantage,
            direction,
            cachedIntersectionList);
    }
    return PickClosestIntersection(cachedIntersectionList, intersection);
}
//-------------------------------------------------------------------------------------------

// Rendering function ver. 1
// where we generate a depth and a normal image.
void DepthScene::Render1(
    size_t pixelsWide,
    size_t pixelsHigh,
    double zoom,
    cv::Mat *depth_img,
    cv::Mat *normal_img) const
{
  const size_t smallerDim =
      ((pixelsWide < pixelsHigh) ? pixelsWide : pixelsHigh);
  const double largeZoom  = zoom * smallerDim;

  // The camera is located at the origin.
  Vector camera(0.0, 0.0, 0.0);

  // The camera faces in the -z direction.
  // This allows the +x direction to be to the right,
  // and the +y direction to be upward.
  Vector direction(0.0, 0.0, -1.0);

  if(depth_img)
  {
    depth_img->create(pixelsHigh, pixelsWide, CV_32FC1);
    (*depth_img)= cv::Scalar::all(0.0);
  }
  if(normal_img)
  {
    normal_img->create(pixelsHigh, pixelsWide, CV_32FC3);
    (*normal_img)= cv::Scalar::all(0.0);
  }

  for (size_t i=0; i < pixelsWide; ++i)
  {
    direction.x = (i - pixelsWide/2.0) / largeZoom;
    for (size_t j=0; j < pixelsHigh; ++j)
    {
      direction.y = (pixelsHigh/2.0 - j) / largeZoom;
      Intersection pixel= TraceRay(camera, direction);
      if(pixel.solid)
      {
        float z= std::sqrt(pixel.distanceSquared);
        Vector &n= pixel.surfaceNormal;
        if(depth_img)
          depth_img->at<float>(j,i)= z;
        if(normal_img)
          // normal_img->at<cv::Vec3f>(j,i)= cv::Vec3f(n.x, n.y, n.z);
          normal_img->at<cv::Vec3f>(j,i)= cv::Vec3f(0.5*(1.0+n.x), 0.5*(1.0+n.y), 0.5*(1.0+n.z));
      }
    }
  }
}
//-------------------------------------------------------------------------------------------

// Rendering function ver. 2
// where we generate a depth and a normal image
// with considering a camera model.
void DepthScene::Render2(
    const TCameraInfo &cam,
    cv::Mat *depth_img,
    cv::Mat *normal_img) const
{
  // The camera is located at the origin.
  Vector camera(0.0, 0.0, 0.0);
  Vector direction(0.0, 0.0, 1.0);

  if(depth_img)
  {
    depth_img->create(cam.Height, cam.Width, CV_32FC1);
    (*depth_img)= cv::Scalar::all(0.0);
  }
  if(normal_img)
  {
    normal_img->create(cam.Height, cam.Width, CV_32FC3);
    (*normal_img)= cv::Scalar::all(0.0);
  }

  for (int i=0; i < cam.Width; ++i)
  {
    for (int j=0; j < cam.Height; ++j)
    {
      cam.InvProject(i, j, direction.x, direction.y);
      Intersection pixel= TraceRay(camera, direction);
      if(pixel.solid)
      {
        float z= std::sqrt(pixel.distanceSquared);
        Vector &n= pixel.surfaceNormal;
        if(depth_img)
          depth_img->at<float>(j,i)= z;
        if(normal_img)
          // normal_img->at<cv::Vec3f>(j,i)= cv::Vec3f(n.x, n.y, n.z);
          // normal_img->at<cv::Vec3f>(j,i)= cv::Vec3f(0.5*(1.0+n.x), 0.5*(1.0+n.y), 0.5*(1.0+n.z));
          // normal_img->at<cv::Vec3f>(j,i)= cv::Vec3f(0.5*(1.0+n.y), 0.5*(1.0+n.x), 0.5*(1.0+n.z));
          normal_img->at<cv::Vec3f>(j,i)= cv::Vec3f(0.5*(1.0-n.x), 0.5*(1.0-n.y), 0.5*(1.0-n.z));
      }
    }
  }
}
//-------------------------------------------------------------------------------------------

// Rendering function ver. 3
// where we generate a depth and a normal image
// with considering a camera model and a region of interest.
// Output images are cropped.
void DepthScene::Render3(
    const TCameraInfo &cam,
    const TROI &roi,
    cv::Mat *depth_img,
    cv::Mat *normal_img) const
{
  // The camera is located at the origin.
  Vector camera(0.0, 0.0, 0.0);
  Vector direction(0.0, 0.0, 1.0);

  int roi_x1(0), roi_y1(0), roi_x2(0), roi_y2(0);
  cam.Project(roi.Cx-1.4142136*roi.Radius, roi.Cy-1.4142136*roi.Radius, roi.Cz, roi_x1, roi_y1);
  cam.Project(roi.Cx+1.4142136*roi.Radius, roi.Cy+1.4142136*roi.Radius, roi.Cz, roi_x2, roi_y2);
  if(roi_x2<roi_x1)  std::swap(roi_x1,roi_x2);
  if(roi_y2<roi_y1)  std::swap(roi_y1,roi_y2);
  // if(roi_x1>=cam.Width)  roi_x1= cam.Width;
  // if(roi_x1<0)           roi_x1= 0;
  // if(roi_x2>=cam.Width)  roi_x2= cam.Width;
  // if(roi_x2<0)           roi_x2= 0;
  // if(roi_y1>=cam.Height) roi_y1= cam.Height;
  // if(roi_y1<0)           roi_y1= 0;
  // if(roi_y2>=cam.Height) roi_y2= cam.Height;
  // if(roi_y2<0)           roi_y2= 0;
  if(roi_x1==roi_x2)  ++roi_x2;
  if(roi_y1==roi_y2)  ++roi_y2;

  // if(depth_img)   depth_img->create(cam.Height, cam.Width, CV_32FC1);
  // if(normal_img)  normal_img->create(cam.Height, cam.Width, CV_32FC3);
  if(depth_img)
  {
    depth_img->create(roi_y2-roi_y1, roi_x2-roi_x1, CV_32FC1);
    (*depth_img)= cv::Scalar::all(0.0);
  }
  if(normal_img)
  {
    normal_img->create(roi_y2-roi_y1, roi_x2-roi_x1, CV_32FC3);
    (*normal_img)= cv::Scalar::all(0.0);
  }

  // Actual image region:
  int aroi_i1(roi_x2-roi_x1), aroi_j1(roi_y2-roi_y1), aroi_i2(0), aroi_j2(0);

  for(int xp=roi_x1; xp<roi_x2; ++xp)
  {
    for(int yp=roi_y1; yp<roi_y2; ++yp)
    {
      cam.InvProject(xp, yp, direction.x, direction.y);
      int i(xp-roi_x1), j(yp-roi_y1);
      Intersection pixel= TraceRay(camera, direction);
      if(pixel.solid)
      {
        if(i<aroi_i1)    aroi_i1= i;
        if(i+1>aroi_i2)  aroi_i2= i+1;
        if(j<aroi_j1)    aroi_j1= j;
        if(j+1>aroi_j2)  aroi_j2= j+1;
        float z= std::sqrt(pixel.distanceSquared);
        Vector &n= pixel.surfaceNormal;
        if(depth_img)
          depth_img->at<float>(j,i)= z;
        if(normal_img)
          // normal_img->at<cv::Vec3f>(j,i)= cv::Vec3f(n.x, n.y, n.z);
          // normal_img->at<cv::Vec3f>(j,i)= cv::Vec3f(0.5*(1.0+n.x), 0.5*(1.0+n.y), 0.5*(1.0+n.z));
          // normal_img->at<cv::Vec3f>(j,i)= cv::Vec3f(0.5*(1.0+n.y), 0.5*(1.0+n.x), 0.5*(1.0+n.z));
          normal_img->at<cv::Vec3f>(j,i)= cv::Vec3f(0.5*(1.0-n.x), 0.5*(1.0-n.y), 0.5*(1.0-n.z));
      }
    }
  }
  cv::Rect aroi(aroi_i1,aroi_j1, aroi_i2-aroi_i1, aroi_j2-aroi_j1);
  if(depth_img)   *depth_img= (*depth_img)(aroi);
  if(normal_img)  *normal_img= (*normal_img)(aroi);
}
//-------------------------------------------------------------------------------------------


// Rendering function ver. 4
// where we generate a raw intersection information.
void DepthScene::Render4(
    const TCameraInfo &cam,
    const TROI &roi,
    std::list<Intersection> &intersection_list,
    int step_xp, int step_yp) const
{
  // The camera is located at the origin.
  Vector camera(0.0, 0.0, 0.0);
  Vector direction(0.0, 0.0, 1.0);

  int roi_x1(0), roi_y1(0), roi_x2(0), roi_y2(0);
  cam.Project(roi.Cx-1.4142136*roi.Radius, roi.Cy-1.4142136*roi.Radius, roi.Cz, roi_x1, roi_y1);
  cam.Project(roi.Cx+1.4142136*roi.Radius, roi.Cy+1.4142136*roi.Radius, roi.Cz, roi_x2, roi_y2);
  if(roi_x2<roi_x1)  std::swap(roi_x1,roi_x2);
  if(roi_y2<roi_y1)  std::swap(roi_y1,roi_y2);
  if(roi_x1==roi_x2)  ++roi_x2;
  if(roi_y1==roi_y2)  ++roi_y2;

  intersection_list.clear();

  for(int xp=roi_x1; xp<roi_x2; xp+=step_xp)
  {
    for(int yp=roi_y1; yp<roi_y2; yp+=step_yp)
    {
      cam.InvProject(xp, yp, direction.x, direction.y);
      Intersection pixel= TraceRay(camera, direction);
      if(pixel.solid)
      {
        intersection_list.push_back(pixel);
      }
    }
  }
}
//-------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------
}  // end of Imager
//-------------------------------------------------------------------------------------------

