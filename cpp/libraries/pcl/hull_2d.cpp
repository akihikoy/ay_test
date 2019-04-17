/*
src:
  http://www.pointclouds.org/documentation/tutorials/hull_2d.php
compile:
  x++ hull_2d.cpp -- -I/home/akihiko/install/boost1.49/include -I/usr/include/eigen3 -I/opt/ros/groovy/include -I/opt/ros/groovy/include/pcl-1.6 -I/usr/include/vtk-5.8 -L/opt/ros/groovy/lib -lpcl_visualization -lpcl_search -lpcl_kdtree -lpcl_filters -lpcl_features -lpcl_segmentation -lpcl_surface -lpcl_io -lpcl_common -lrostime -lvtkCommon -lvtkFiltering -lvtkRendering -lvtkGraphics -lboost_system-mt -lboost_thread-mt -Wl,-rpath /opt/ros/groovy/lib
special:
  -lpcl_surface
usage:
  Construct a concave hull and visualize it
  ./a.out
  ./a.out POINT_CLOUD.pcd
*/
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/surface/concave_hull.h>
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


  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>),
                                      cloud_projected (new pcl::PointCloud<pcl::PointXYZ>);
  // Build a filter to remove spurious NaNs
  pcl::PassThrough<pcl::PointXYZ> pass;
  pass.setInputCloud (cloud);
  pass.setFilterFieldName ("z");
  pass.setFilterLimits (0, 1.1);
  pass.filter (*cloud_filtered);
  std::cerr << "PointCloud after filtering has: "
            << cloud_filtered->points.size () << " data points." << std::endl;

  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
  // Create the segmentation object
  pcl::SACSegmentation<pcl::PointXYZ> seg;
  // Optional
  seg.setOptimizeCoefficients (true);
  // Mandatory
  seg.setModelType (pcl::SACMODEL_PLANE);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setDistanceThreshold (0.01);

  seg.setInputCloud (cloud_filtered);
  seg.segment (*inliers, *coefficients);
  std::cerr << "PointCloud after segmentation has: "
            << inliers->indices.size () << " inliers." << std::endl;

  // Project the model inliers
  pcl::ProjectInliers<pcl::PointXYZ> proj;
  proj.setModelType (pcl::SACMODEL_PLANE);
  proj.setIndices (inliers);
  proj.setInputCloud (cloud_filtered);
  proj.setModelCoefficients (coefficients);
  proj.filter (*cloud_projected);
  std::cerr << "PointCloud after projection has: "
            << cloud_projected->points.size () << " data points." << std::endl;

  // Create a Concave Hull representation of the projected inliers
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_hull (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::ConcaveHull<pcl::PointXYZ> chull;
  chull.setInputCloud (cloud_projected);
  chull.setAlpha (0.1);
  // chull.setAlpha (0.005);
  chull.reconstruct (*cloud_hull);

  std::cerr << "Concave hull has: " << cloud_hull->points.size ()
            << " data points." << std::endl;

  // pcl::PCDWriter writer;
  // writer.write ("data/table_scene_mug_stereo_textured_hull.pcd", *cloud_hull, false);



  // Visualize
  float gray1_rgb[3]= {64,64,64};
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
  // viewer= SetupViewer(cloud_org.makeShared());
  viewer= SetupViewer(ColorPointCloud(cloud,gray1_rgb));

  float gray2_rgb[3]= {128,128,128};
  AddCloud(viewer,ColorPointCloud(cloud_projected,gray2_rgb),GetID("cloud_projected",1),2);

  // viewer->addPolygonMesh(triangles);
  float rgb[3]= {0,0,255};
  AddCloud(viewer,ColorPointCloud(cloud_hull,rgb),GetID("cloud_hull",1),5);

  viewer->addPolygon<pcl::PointXYZ>(cloud_hull, 1.0, 0.5, 0.5, GetID("polygon_hull",1));

  while(!viewer->wasStopped())  {viewer->spinOnce(100);}

  return (0);
}
