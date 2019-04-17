/*
src:
  http://pointclouds.org/documentation/tutorials/random_sample_consensus.php
compile:
  x++ ransac2.cpp -- -I/home/akihiko/install/boost1.49/include -I/usr/include/eigen3 -I/opt/ros/groovy/include -I/opt/ros/groovy/include/pcl-1.6 -I/usr/include/vtk-5.8 -L/opt/ros/groovy/lib -lpcl_visualization -lpcl_sample_consensus -lpcl_segmentation -lpcl_common -lpcl_io -lvtkCommon -lvtkFiltering -Wl,-rpath /opt/ros/groovy/lib
usage:
  RANSAC test (plane/sphere)
  ./a.out
*/
#include <iostream>
#include <pcl/console/parse.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/ransac.h>
#if 1
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/sac_model_sphere.h>
// #include <pcl/sample_consensus/sac_model_cylinder.h>
#endif
#if 0
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#endif
// #include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>

// boost::shared_ptr<pcl::visualization::PCLVisualizer>
// simpleVis (pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud)
// {
  // // --------------------------------------------
  // // -----Open 3D viewer and add point cloud-----
  // // --------------------------------------------
  // boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  // viewer->setBackgroundColor (0, 0, 0);
  // viewer->addPointCloud<pcl::PointXYZ> (cloud, "sample cloud");
  // viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
  // //viewer->addCoordinateSystem (1.0, "global");
  // viewer->initCameraParameters ();
  // return (viewer);
// }

int
main(int argc, char** argv)
{
  // initialize PointClouds
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr final (new pcl::PointCloud<pcl::PointXYZ>);

  // populate our PointCloud with points
  cloud->width    = 500;
  cloud->height   = 1;
  cloud->is_dense = false;
  cloud->points.resize (cloud->width * cloud->height);
  for (size_t i = 0; i < cloud->points.size (); ++i)
  {
    if (pcl::console::find_argument (argc, argv, "-s") >= 0 || pcl::console::find_argument (argc, argv, "-sf") >= 0)
    {
      cloud->points[i].x = 1024 * rand () / (RAND_MAX + 1.0);
      cloud->points[i].y = 1024 * rand () / (RAND_MAX + 1.0);
      if (i % 5 == 0)
        cloud->points[i].z = 1024 * rand () / (RAND_MAX + 1.0);
      else if(i % 2 == 0)
        cloud->points[i].z =  sqrt( 1 - (cloud->points[i].x * cloud->points[i].x)
                                      - (cloud->points[i].y * cloud->points[i].y));
      else
        cloud->points[i].z =  - sqrt( 1 - (cloud->points[i].x * cloud->points[i].x)
                                        - (cloud->points[i].y * cloud->points[i].y));
    }
    else
    {
      cloud->points[i].x = 1024 * rand () / (RAND_MAX + 1.0);
      cloud->points[i].y = 1024 * rand () / (RAND_MAX + 1.0);
      if( i % 2 == 0)
        cloud->points[i].z = 1024 * rand () / (RAND_MAX + 1.0);
      else
        cloud->points[i].z = -1 * (cloud->points[i].x + cloud->points[i].y);
    }
  }

#if 1
  std::vector<int> inliers;

  // created RandomSampleConsensus object and compute the appropriated model
  pcl::SampleConsensusModelSphere<pcl::PointXYZ>::Ptr
    model_s(new pcl::SampleConsensusModelSphere<pcl::PointXYZ> (cloud));
  pcl::SampleConsensusModelPlane<pcl::PointXYZ>::Ptr
    model_p (new pcl::SampleConsensusModelPlane<pcl::PointXYZ> (cloud));
  // pcl::SampleConsensusModelCylinder<pcl::PointXYZ, pcl::Normal>::Ptr
  //   model_c (new pcl::SampleConsensusModelCylinder<pcl::PointXYZ, pcl::Normal> (cloud));
  if(pcl::console::find_argument (argc, argv, "-f") >= 0)
  {
    pcl::RandomSampleConsensus<pcl::PointXYZ> ransac (model_p);
    ransac.setDistanceThreshold (.01);
    ransac.computeModel();
    ransac.getInliers(inliers);
  }
  else if (pcl::console::find_argument (argc, argv, "-sf") >= 0 )
  {
    pcl::RandomSampleConsensus<pcl::PointXYZ> ransac (model_s);
    ransac.setDistanceThreshold (.01);
    ransac.computeModel();
    ransac.getInliers(inliers);
  }
  // else if (pcl::console::find_argument (argc, argv, "-cf") >= 0 )
  // {
    // pcl::RandomSampleConsensus<pcl::PointXYZ> ransac (model_c);
    // ransac.setDistanceThreshold (.01);
    // ransac.computeModel();
    // ransac.getInliers(inliers);
  // }

  std::cerr << "Model inliers: " << inliers.size () << std::endl;

  // copies all inliers of the model computed to another PointCloud
  pcl::copyPointCloud<pcl::PointXYZ>(*cloud, inliers, *final);
#endif

#if 0
  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
  // Create the segmentation object
  pcl::SACSegmentation<pcl::PointXYZ> seg;
  // Optional
  seg.setOptimizeCoefficients (true);
  // Mandatory

  if(pcl::console::find_argument (argc, argv, "-f") >= 0)
  {
    seg.setModelType (pcl::SACMODEL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setDistanceThreshold (0.1);

    seg.setInputCloud (cloud);  // .makeShared () ?
    seg.segment (*inliers, *coefficients);

    if (inliers->indices.size () == 0)
    {
      PCL_ERROR ("Could not estimate a planar model for the given dataset.");
      return (-1);
    }
  }
  else if (pcl::console::find_argument (argc, argv, "-sf") >= 0 )
  {
    seg.setModelType (pcl::SACMODEL_SPHERE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setDistanceThreshold (0.1);

    seg.setInputCloud (cloud);  // .makeShared () ?
    seg.segment (*inliers, *coefficients);

    if (inliers->indices.size () == 0)
    {
      PCL_ERROR ("Could not estimate a planar model for the given dataset.");
      return (-1);
    }
  }

  std::cerr << "Model inliers: " << inliers->indices.size () << std::endl;

  // copies all inliers of the model computed to another PointCloud
  pcl::copyPointCloud<pcl::PointXYZ>(*cloud, *inliers, *final);
#endif


#if 0
  // creates the visualization object and adds either our orignial cloud or all of the inliers
  // depending on the command line arguments specified.
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
  if (pcl::console::find_argument (argc, argv, "-f") >= 0 || pcl::console::find_argument (argc, argv, "-sf") >= 0)
    viewer = simpleVis(final);
  else
    viewer = simpleVis(cloud);
  while (!viewer->wasStopped ())
  {
    viewer->spinOnce(100);
    usleep(100000);
  }
#endif

#if 1
  pcl::visualization::CloudViewer viewer("Cloud Viewer");
  if (pcl::console::find_argument(argc, argv, "-f") >= 0 || pcl::console::find_argument(argc, argv, "-sf") >= 0 || pcl::console::find_argument(argc, argv, "-cf") >= 0)
    viewer.showCloud(final);
  else
    viewer.showCloud(cloud);
  while (!viewer.wasStopped ()) {}
#endif


  return 0;
 }

