//-------------------------------------------------------------------------------------------
/*! \file    depthscene.h
    \brief   Depth scene for ray tracing using Don Cross's library (doncross).
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Feb.03, 2015
*/
//-------------------------------------------------------------------------------------------
#ifndef depthscene_h
#define depthscene_h
//-------------------------------------------------------------------------------------------
#include "doncross/raytrace/raytrace/imager.h"
#include <opencv2/core/core.hpp>
//-------------------------------------------------------------------------------------------
namespace Imager
{
//-------------------------------------------------------------------------------------------

// Extension of Don Cross's library:

// Wrap a SolidObject to change the center of rotation.
class SolidObject_Wrapper: public SolidObject
{
public:
    explicit SolidObject_Wrapper(const Vector& _center, SolidObject* _other)
        : SolidObject(_center)
        , other(_other)
    {
        SetTag("SolidObject_Wrapper");
    }

    virtual ~SolidObject_Wrapper()
    {
        delete other;
        other = NULL;
    }

    virtual bool Contains(const Vector& point) const
    {
        return other->Contains(point);
    }

    virtual void AppendAllIntersections(
        const Vector& vantage,
        const Vector& direction,
        IntersectionList& intersectionList) const
    {
        other->AppendAllIntersections(vantage, direction, intersectionList);
    }

    virtual SolidObject& Translate(double dx, double dy, double dz)
    {
        SolidObject::Translate(dx, dy, dz);
        other->Translate(dx, dy, dz);
        return *this;
    }

    virtual SolidObject& RotateX(double angleInDegrees)
    {
        const double angleInRadians = RadiansFromDegrees(angleInDegrees);
        const double a = cos(angleInRadians);
        const double b = sin(angleInRadians);

        NestedRotateX(*other,  angleInDegrees, a, b);

        return *this;
    }

    virtual SolidObject& RotateY(double angleInDegrees)
    {
        const double angleInRadians = RadiansFromDegrees(angleInDegrees);
        const double a = cos(angleInRadians);
        const double b = sin(angleInRadians);

        NestedRotateY(*other,  angleInDegrees, a, b);

        return *this;
    }

    virtual SolidObject& RotateZ(double angleInDegrees)
    {
        const double angleInRadians = RadiansFromDegrees(angleInDegrees);
        const double a = cos(angleInRadians);
        const double b = sin(angleInRadians);

        NestedRotateZ(*other,  angleInDegrees, a, b);

        return *this;
    }

protected:
    // Copied from SolidObject_BinaryOperator
    void NestedRotateX(SolidObject &nested, double angleInDegrees, double a, double b)
    {
        // Rotate the nested object about its own center.
        nested.RotateX(angleInDegrees);

        // Revolve the center of the nested object around the common center of this binary operator.
        const Vector& c = Center();
        const Vector& nc = nested.Center();
        const double dy = nc.y - c.y;
        const double dz = nc.z - c.z;
        nested.Move (nc.x, c.y + (a*dy - b*dz), c.z + (a*dz + b*dy));
    }

    // Copied from SolidObject_BinaryOperator
    void NestedRotateY(SolidObject &nested, double angleInDegrees, double a, double b)
    {
        // Rotate the nested object about its own center.
        nested.RotateY(angleInDegrees);

        // Revolve the center of the nested object around the common center of this binary operator.
        const Vector& c = Center();
        const Vector& nc = nested.Center();
        const double dx = nc.x - c.x;
        const double dz = nc.z - c.z;
        nested.Move (c.x + (a*dx + b*dz), nc.y, c.z + (a*dz - b*dx));
    }

    // Copied from SolidObject_BinaryOperator
    void NestedRotateZ(SolidObject &nested, double angleInDegrees, double a, double b)
    {
        // Rotate the nested object about its own center.
        nested.RotateZ(angleInDegrees);

        // Revolve the center of the nested object around the common center of this binary operator.
        const Vector& c = Center();
        const Vector& nc = nested.Center();
        const double dx = nc.x - c.x;
        const double dy = nc.y - c.y;
        nested.Move (c.x + (a*dx - b*dy), c.y + (a*dy + b*dx), nc.z);
    }

private:
    SolidObject* other;
};
//-------------------------------------------------------------------------------------------



/* Camera information.
  We assume a simple projection model.
  Let
        [Fx  0  Cx]
    P = [ 0  Fy Cy]
        [ 0  0   1]
  a projection matrix.  Camera is at [0,0,0] and the image plane is z=1.
  A 3D point [xc,yc,zc]^T is projected onto an image plane [xp,yp] by:
    [u,v,w]^T= P * [xc,yc,zc]^T
    xp= u/w
    yp= v/w
*/
struct TCameraInfo
{
  int Width;
  int Height;
  double Fx, Fy;  // focal lengths
  double Cx, Cy;  // principal point

  // Project 3D point [xc,yc,zc] on an image [xp,yp]
  void Project(const double &xc, const double &yc, const double &zc, int &xp, int &yp) const
    {
      xp= Fx*xc/zc + Cx;
      yp= Fy*yc/zc + Cy;
    }

  // Inverse project a point [xp,yp] of image to 3D point [xc,yc,1]
  void InvProject(int xp, int yp, double &xc, double &yc) const
    {
      xc= (xp-Cx)/Fx;
      yc= (yp-Cy)/Fy;
    }

  bool IsInvalid(int xp, int yp)
    {return xp<0 || xp>=Width || yp<0 || yp>=Height;}
  bool IsValid(int xp, int yp)
    {return !IsInvalid(xp,yp);}
};
//-------------------------------------------------------------------------------------------

// Region of interest in 3D.
struct TROI
{
  double Cx, Cy, Cz;  // Center
  double Radius;  // Radius
};
//-------------------------------------------------------------------------------------------


//===========================================================================================
// The depth scene object that renders a depth image of objects
class DepthScene
//===========================================================================================
{
public:
  explicit DepthScene(const Intersection& _backgroundIntersection = Intersection())
      : backgroundIntersection(_backgroundIntersection)
  {
  }

  virtual ~DepthScene()
  {
    ClearSolidObjectList();
  }

  // Caller must allocate solidObject via operator new.
  // This class will then own the responsibility of deleting it.
  SolidObject& AddSolidObject(SolidObject* solidObject)
  {
    solidObjectList.push_back(solidObject);
    return *solidObject;
  }

  int NumSolidObjects() const {return solidObjectList.size();}
  SolidObject* RefSolidObject(int idx)  {return solidObjectList[idx];}
  const SolidObject* RefSolidObject(int idx) const {return solidObjectList[idx];}

  // Rendering function ver. 1
  // where we generate a depth and a normal image.
  void Render1(
      size_t pixelsWide,
      size_t pixelsHigh,
      double zoom,
      cv::Mat *depth_img=NULL,
      cv::Mat *normal_img=NULL) const;

  // Rendering function ver. 2
  // where we generate a depth and a normal image
  // with considering a camera model.
  void Render2(
      const TCameraInfo &cam,
      cv::Mat *depth_img=NULL,
      cv::Mat *normal_img=NULL) const;

  // Rendering function ver. 3
  // where we generate a depth and a normal image
  // with considering a camera model and a region of interest.
  // Output images are cropped.
  void Render3(
      const TCameraInfo &cam,
      const TROI &roi,
      cv::Mat *depth_img=NULL,
      cv::Mat *normal_img=NULL) const;

  // Rendering function ver. 4
  // where we generate a raw intersection information.
  void Render4(
      const TCameraInfo &cam,
      const TROI &roi,
      std::list<Intersection> &intersection_list,
      int step_xp=1, int step_yp=1) const;

private:
  void ClearSolidObjectList();

  int FindClosestIntersection(
      const Vector& vantage,
      const Vector& direction,
      Intersection& intersection) const;

  Intersection TraceRay(
      const Vector& vantage,
      const Vector& direction) const;

  // The intersection to use for pixels where no solid
  // object intersection was found.
  Intersection backgroundIntersection;

  // Define some list types used by member variables below.
  typedef std::vector<SolidObject*> SolidObjectList;

  // A list of all the solid objects in the scene.
  SolidObjectList solidObjectList;

  // Help performance by avoiding constant construction/destruction
  // of intersection lists.
  mutable IntersectionList cachedIntersectionList;
};
//-------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------
}  // end of Imager
//-------------------------------------------------------------------------------------------
#endif // depthscene_h
//-------------------------------------------------------------------------------------------
