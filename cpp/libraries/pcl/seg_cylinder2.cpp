/*
src:
  http://pointclouds.org/documentation/tutorials/cylinder_segmentation.php
compile:
  x++ seg_cylinder2.cpp -- -I/home/akihiko/install/boost1.49/include -I/usr/include/eigen3 -I/opt/ros/groovy/include -I/opt/ros/groovy/include/pcl-1.6 -I/usr/include/vtk-5.8 -L/opt/ros/groovy/lib -lpcl_visualization -lpcl_search -lpcl_kdtree -lpcl_filters -lpcl_features -lpcl_segmentation -lpcl_io -lpcl_common -lrostime -Wl,-rpath /opt/ros/groovy/lib
usage:
  Cylinder segmentation example with visualization
  ./a.out
  ./a.out POINT_CLOUD.pcd
*/
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/features/normal_3d.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/cloud_viewer.h>

typedef pcl::PointXYZ PointT;

int main (int argc, char** argv)
{
  pcl::PCDWriter writer;

  // Read in the cloud data
  pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);
  pcl::PointCloud<pcl::PointXYZRGB> cloud_org;
  if(argc==1)
    pcl::io::loadPCDFile("table_scene_mug_stereo_textured.pcd", cloud_org);
  else
    pcl::io::loadPCDFile(argv[1], cloud_org);
  pcl::copyPointCloud(cloud_org,*cloud);
  std::cerr << "PointCloud has: " << cloud->points.size () << " data points." << std::endl;

  // Build a passthrough filter to remove spurious NaNs
  pcl::PointCloud<PointT>::Ptr cloud_filtered (new pcl::PointCloud<PointT>);
  pcl::PassThrough<PointT> pass;
  pass.setInputCloud (cloud);
  pass.setFilterFieldName ("z");
  // pass.setFilterLimits (0, 1.5);
  pass.setFilterLimits (-100.0, 100.0);
  pass.filter (*cloud_filtered);
  std::cerr << "PointCloud after filtering has: " << cloud_filtered->points.size () << " data points." << std::endl;

  // Estimate point normals
  pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
  pcl::NormalEstimation<PointT, pcl::Normal> ne;
  pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT> ());
  ne.setSearchMethod (tree);
  ne.setInputCloud (cloud_filtered);
  ne.setKSearch (50);
  ne.compute (*cloud_normals);

  // Create the segmentation object for the planar model and set all the parameters
  pcl::SACSegmentationFromNormals<PointT, pcl::Normal> seg;
  pcl::ModelCoefficients::Ptr coefficients_plane(new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers_plane(new pcl::PointIndices);
  seg.setOptimizeCoefficients (true);
  seg.setModelType (pcl::SACMODEL_NORMAL_PLANE);
  seg.setNormalDistanceWeight (0.1);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setMaxIterations (100);
  // seg.setDistanceThreshold (0.03);
  seg.setDistanceThreshold (0.05);
  seg.setInputCloud (cloud_filtered);
  seg.setInputNormals (cloud_normals);
  // Obtain the plane inliers and coefficients
  seg.segment (*inliers_plane, *coefficients_plane);
  std::cerr << "Plane coefficients: " << *coefficients_plane << std::endl;

  // Extract the planar inliers from the input cloud
  pcl::ExtractIndices<PointT> extract;
  extract.setInputCloud (cloud_filtered);
  extract.setIndices (inliers_plane);
  extract.setNegative (false);

  // Write the planar inliers to disk
  pcl::PointCloud<PointT>::Ptr cloud_plane (new pcl::PointCloud<PointT> ());
  extract.filter (*cloud_plane);
  std::cerr << "PointCloud representing the planar component: " << cloud_plane->points.size () << " data points." << std::endl;
  writer.write ("data/table_scene_mug_stereo_textured_plane.pcd", *cloud_plane, false);

  // Remove the planar inliers, extract the rest
  pcl::PointCloud<PointT>::Ptr cloud_filtered2 (new pcl::PointCloud<PointT>);
  pcl::PointCloud<pcl::Normal>::Ptr cloud_normals2 (new pcl::PointCloud<pcl::Normal>);
  extract.setNegative (true);
  extract.filter (*cloud_filtered2);
  pcl::ExtractIndices<pcl::Normal> extract_normals;
  extract_normals.setNegative (true);
  extract_normals.setInputCloud (cloud_normals);
  extract_normals.setIndices (inliers_plane);
  extract_normals.filter (*cloud_normals2);

  // Create the segmentation object for cylinder segmentation and set all the parameters
  // We will reuse the previous seg
  pcl::ModelCoefficients::Ptr coefficients_cylinder(new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers_cylinder(new pcl::PointIndices);
  seg.setOptimizeCoefficients (true);
  seg.setModelType (pcl::SACMODEL_CYLINDER);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setNormalDistanceWeight (0.1);
  seg.setMaxIterations (10000);
  seg.setDistanceThreshold (0.05);
  seg.setRadiusLimits (0, 0.1);
  seg.setInputCloud (cloud_filtered2);
  seg.setInputNormals (cloud_normals2);
  // Obtain the cylinder inliers and coefficients
  seg.segment (*inliers_cylinder, *coefficients_cylinder);
  std::cerr << "Cylinder coefficients: " << *coefficients_cylinder << std::endl;

  // Write the cylinder inliers to disk
  extract.setInputCloud (cloud_filtered2);
  extract.setIndices (inliers_cylinder);
  extract.setNegative (false);
  pcl::PointCloud<PointT>::Ptr cloud_cylinder (new pcl::PointCloud<PointT> ());
  extract.filter (*cloud_cylinder);
  if (cloud_cylinder->points.empty ())
    std::cerr << "Can't find the cylindrical component." << std::endl;
  else
  {
    std::cerr << "PointCloud representing the cylindrical component: " << cloud_cylinder->points.size () << " data points." << std::endl;
    writer.write ("data/table_scene_mug_stereo_textured_cylinder.pcd", *cloud_cylinder, false);
  }


  // Visualization: cloud_org + cloud_plane + cloud_cylinder:
  pcl::visualization::CloudViewer viewer("cluster viewer");
  pcl::PointCloud<pcl::PointXYZRGB> cloud_c, cloud_plane_col, cloud_cylinder_col;

  float colors[6][3] ={{255, 0, 0}, {0,255,0}, {0,0,255}, {255,255,0}, {0,255,255}, {255,0,255}};
  int j(0);
  for(size_t i = 0; i < cloud_org.points.size (); ++i)
  {
    cloud_org.points[i].r = 255;
    cloud_org.points[i].g = 255;
    cloud_org.points[i].b = 255;
  }
  j= 0;
  pcl::copyPointCloud(*cloud_plane, cloud_plane_col);
  for(size_t i = 0; i < cloud_plane_col.points.size (); ++i)
  {
    cloud_plane_col.points[i].r = colors[j%6][0];
    cloud_plane_col.points[i].g = colors[j%6][1];
    cloud_plane_col.points[i].b = colors[j%6][2];
  }
  j= 1;
  pcl::copyPointCloud(*cloud_cylinder, cloud_cylinder_col);
  for(size_t i = 0; i < cloud_cylinder_col.points.size (); ++i)
  {
    cloud_cylinder_col.points[i].r = colors[j%6][0];
    cloud_cylinder_col.points[i].g = colors[j%6][1];
    cloud_cylinder_col.points[i].b = colors[j%6][2];
  }

  cloud_c= cloud_org;
  cloud_c+= cloud_plane_col;
  cloud_c+= cloud_cylinder_col;
  viewer.showCloud (cloud_c.makeShared());
  while (!viewer.wasStopped ()) {}

  return (0);
}

