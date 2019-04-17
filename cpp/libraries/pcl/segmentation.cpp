/*
src:
  http://ros-robot.blogspot.com/2011/08/pclapi-point-cloud-library-pcl-pcl-api.html
  http://pointclouds.org/documentation/tutorials/planar_segmentation.php
compile:
  x++ segmentation.cpp -- -I/home/akihiko/install/boost1.49/include -I/usr/include/eigen3 -I/opt/ros/groovy/include -I/opt/ros/groovy/include/pcl-1.6 -I/usr/include/vtk-5.8 -L/opt/ros/groovy/lib -lpcl_visualization -lpcl_segmentation -lpcl_common -Wl,-rpath /opt/ros/groovy/lib
usage:
  RANSAC segmentation test
  ./a.out
*/
#include <iostream>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/cloud_viewer.h>

int main (int argc, char** argv)
{
  pcl::PointCloud<pcl::PointXYZRGB> cloud;

  // Fill in the cloud data
  cloud.width  = 15;
  cloud.height = 10;
  cloud.points.resize (cloud.width * cloud.height);

  {
    size_t i;
    for (i = 0; i < cloud.points.size ()/2; ++i)
    {
      // Put on a plane
      cloud.points[i].x = 1024 * rand () / (RAND_MAX + 1.0);
      cloud.points[i].y = 1024 * rand () / (RAND_MAX + 1.0);
      cloud.points[i].z = 0.1 * rand () / (RAND_MAX + 1.0);
      cloud.points[i].r = 255;
      cloud.points[i].g = 255;
      cloud.points[i].b = 255;
    }
    for (; i < cloud.points.size (); ++i)
    {
      // Randomly
      cloud.points[i].x = 1024 * rand () / (RAND_MAX + 1.0);
      cloud.points[i].y = 1024 * rand () / (RAND_MAX + 1.0);
      cloud.points[i].z = 1024 * rand () / (RAND_MAX + 1.0);
      cloud.points[i].r = 255;
      cloud.points[i].g = 255;
      cloud.points[i].b = 255;
    }
  }

  // Set a few outliers
  cloud.points[0].z = 2.0;
  cloud.points[3].z = -2.0;
  cloud.points[6].z = 4.0;

  std::cerr << "Point cloud data: " << cloud.points.size () << " points" << std::endl;
  for (size_t i = 0; i < cloud.points.size (); ++i)
    std::cerr << "    " << cloud.points[i].x << " "
      << cloud.points[i].y << " "
      << cloud.points[i].z << std::endl;

  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
  // Create the segmentation object
  pcl::SACSegmentation<pcl::PointXYZRGB> seg;
  // Optional
  seg.setOptimizeCoefficients (true);
  // Mandatory
  seg.setModelType (pcl::SACMODEL_PLANE);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setDistanceThreshold (0.1);

  seg.setInputCloud (cloud.makeShared ());
  seg.segment (*inliers, *coefficients);

  if (inliers->indices.size () == 0)
  {
    PCL_ERROR ("Could not estimate a planar model for the given dataset.");
    return (-1);
  }

  std::cerr << "Model coefficients: " << coefficients->values[0] << " "
    << coefficients->values[1] << " "
    << coefficients->values[2] << " "
    << coefficients->values[3] << std::endl;

  std::cerr << "Model inliers: " << inliers->indices.size () << std::endl;
  for (size_t i = 0; i < inliers->indices.size (); ++i)
  {
    std::cerr << inliers->indices[i] << "    " << cloud.points[inliers->indices[i]].x << " "
      << cloud.points[inliers->indices[i]].y << " "
      << cloud.points[inliers->indices[i]].z << std::endl;
    cloud.points[inliers->indices[i]].r = 255;
    cloud.points[inliers->indices[i]].g = 0;
    cloud.points[inliers->indices[i]].b = 0;
  }
  pcl::visualization::CloudViewer viewer("Cloud Viewer");
  viewer.showCloud(cloud.makeShared());
  while (!viewer.wasStopped ()) {}

  return (0);
}
