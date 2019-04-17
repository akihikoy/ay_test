/*
src:
  http://pointclouds.org/documentation/tutorials/resampling.php
compile:
  x++ smoothing2.cpp -- -I/home/akihiko/install/boost1.49/include -I/usr/include/eigen3 -I/opt/ros/groovy/include -I/opt/ros/groovy/include/pcl-1.6 -I/usr/include/vtk-5.8 -L/opt/ros/groovy/lib -lpcl_visualization -lpcl_search -lpcl_kdtree -lpcl_filters -lpcl_features -lpcl_segmentation -lpcl_surface -lpcl_io -lpcl_common -lrostime -lvtkCommon -lvtkFiltering -lvtkRendering -lvtkGraphics -lboost_system-mt -lboost_thread-mt -Wl,-rpath /opt/ros/groovy/lib
special:
  -lpcl_surface
usage:
  Smoothing a surface
  ./a.out
  ./a.out POINT_CLOUD.pcd
*/

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/surface/mls.h>
#include <pcl/filters/voxel_grid.h>

#include <pcl/visualization/cloud_viewer.h>

inline std::string GetID(const std::string &base, int idx)
{
  std::stringstream ss;
  ss<<base<<idx;
  return ss.str();
}

boost::shared_ptr<pcl::visualization::PCLVisualizer>
SetupViewer(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud)
{
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  viewer->addCoordinateSystem (0.2);

  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
  viewer->addPointCloud<pcl::PointXYZRGB> (cloud, rgb, "cloud1");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud1");

  viewer->initCameraParameters ();
  viewer->resetCameraViewpoint("cloud1");
  return viewer;
}

void AddCloud(
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer,
    pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud,
    const std::string &name, int psize=1)
{
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
  viewer->addPointCloud<pcl::PointXYZRGB> (cloud, rgb, name);
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, psize, name);
}

void ColorPointCloud(
    const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &cloud_in,
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud_out,
    const float rgb[3])
{
  pcl::copyPointCloud(*cloud_in, *cloud_out);
  for (int i(0); i<cloud_out->points.size(); ++i)
  {
    cloud_out->points[i].r = rgb[0];
    cloud_out->points[i].g = rgb[1];
    cloud_out->points[i].b = rgb[2];
  }
}
pcl::PointCloud<pcl::PointXYZRGB>::Ptr ColorPointCloud(
    const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &cloud_in,
    const float rgb[3])
{
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr  cloud_out(new pcl::PointCloud<pcl::PointXYZRGB>);
  ColorPointCloud(cloud_in, cloud_out, rgb);
  return cloud_out;
}


typedef pcl::PointXYZ PointT;

int main (int argc, char** argv)
{
  // Read in the cloud data
  pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);
  pcl::PointCloud<pcl::PointXYZRGB> cloud_org;
  if(argc==1)
    pcl::io::loadPCDFile("table_scene_mug_stereo_textured.pcd", cloud_org);
  else
    pcl::io::loadPCDFile(argv[1], cloud_org);
  pcl::copyPointCloud(cloud_org,*cloud);
  std::cerr << "PointCloud has: " << cloud->points.size () << " data points." << std::endl;


  // Create the filtering object: downsample the dataset using a leaf size of 0.1cm
  pcl::VoxelGrid<pcl::PointXYZ> vg;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
  vg.setInputCloud (cloud);
  // vg.setLeafSize (0.01, 0.01, 0.01);
  vg.setLeafSize (0.001, 0.001, 0.001);
  vg.filter (*cloud_filtered);
  std::cout << "PointCloud after filtering has: " << cloud_filtered->points.size ()  << " data points." << std::endl; //*



  // Create a KD-Tree
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);

  pcl::PointCloud<pcl::PointXYZ> mls_points;

  pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointXYZ> mls;

  mls.setComputeNormals (false);

  // Set parameters
  mls.setInputCloud (cloud_filtered);
  mls.setPolynomialFit (true);
  mls.setSearchMethod (tree);
  mls.setSearchRadius (0.01);

  // Reconstruct
  mls.process (mls_points);

  // Save output
  pcl::io::savePCDFile ("data/smoothed-mls.pcd", mls_points);



  pcl::PointCloud<PointT>::Ptr cloud_mls (new pcl::PointCloud<PointT>);
  pcl::copyPointCloud(mls_points, *cloud_mls);

  // Visualize
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;

  // viewer->addPolygonMesh(triangles);
  const float rgb_mls[3]= {0,0,255};
  // viewer= SetupViewer(cloud_org.makeShared());
  // AddCloud(viewer,ColorPointCloud(cloud_mls,rgb_mls),"cloud_mls",4);
  viewer= SetupViewer(ColorPointCloud(cloud_mls,rgb_mls));

  while(!viewer->wasStopped())  {viewer->spinOnce(100);}

  return (0);
}
