/*
refs:
  http://ros-robot.blogspot.com/2011/08/point-cloud-library_25.html
  http://pointclouds.org/documentation/tutorials/cluster_extraction.php
  http://pointclouds.org/documentation/tutorials/kdtree_search.php
  http://docs.pointclouds.org/trunk/classpcl_1_1_kd_tree.html
compile:
  x++ cluster2.cpp -- -I/home/akihiko/install/boost1.49/include -I/usr/include/eigen3 -I/opt/ros/groovy/include -I/opt/ros/groovy/include/pcl-1.6 -I/usr/include/vtk-5.8 -L/opt/ros/groovy/lib -lpcl_visualization -lpcl_segmentation -lpcl_io -lpcl_kdtree -lpcl_filters -lrostime -lpcl_common -Wl,-rpath /opt/ros/groovy/lib
usage:
  Extract clusters from a point cloud; obtained points are as high resolution as the original point cloud
  ./a.out
  ./a.out POINT_CLOUD.pcd
*/
#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/visualization/cloud_viewer.h>

int main (int argc, char** argv)
{
  // Read in the cloud data
  pcl::PointCloud<pcl::PointXYZRGB> cloud_org;
  if(argc==1)
    pcl::io::loadPCDFile("table_scene_lms400.pcd", cloud_org);
  else
    pcl::io::loadPCDFile(argv[1], cloud_org);

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::copyPointCloud(cloud_org,*cloud);
  std::cout << "PointCloud before filtering has: " << cloud->points.size () << " data points." << std::endl; //*

  // Create the filtering object: downsample the dataset using a leaf size of 1cm
  pcl::VoxelGrid<pcl::PointXYZ> vg;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
  vg.setInputCloud (cloud);
  // vg.setLeafSize (0.01, 0.01, 0.01);
  vg.setLeafSize (0.005, 0.005, 0.005);
  vg.filter (*cloud_filtered);
  std::cout << "PointCloud after filtering has: " << cloud_filtered->points.size ()  << " data points." << std::endl; //*

  // Create the segmentation object for the planar model and set all the parameters
  pcl::SACSegmentation<pcl::PointXYZ> seg;
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane (new pcl::PointCloud<pcl::PointXYZ> ());
  pcl::PCDWriter writer;
  seg.setOptimizeCoefficients (true);
  seg.setModelType (pcl::SACMODEL_PLANE);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setMaxIterations (100);
  seg.setDistanceThreshold (0.02);

  int nr_points = cloud_filtered->points.size ();
  while (cloud_filtered->points.size () > 0.3 * nr_points)
  {
    // Segment the largest planar component from the remaining cloud
    seg.setInputCloud(cloud_filtered);
    seg.segment (*inliers, *coefficients); //*
    if (inliers->indices.size () == 0)
    {
      std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
      break;
    }

    // Extract the planar inliers from the input cloud
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud (cloud_filtered);
    extract.setIndices (inliers);
    extract.setNegative (false);

    // Write the planar inliers to disk
    extract.filter (*cloud_plane); //*
    std::cout << "PointCloud representing the planar component: " << cloud_plane->points.size () << " data points." << std::endl;

    // Remove the planar inliers, extract the rest
    extract.setNegative (true);
    extract.filter (*cloud_filtered); //*
  }

  // Creating the KdTree object for the search method of the extraction
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
  tree->setInputCloud (cloud_filtered);

  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
  ec.setClusterTolerance (0.02); // 2cm
  ec.setMinClusterSize (100);
  ec.setMaxClusterSize (25000);
  ec.setSearchMethod (tree);
  ec.setInputCloud (cloud_filtered);
  ec.extract (cluster_indices);

  int j = 0;
  float colors[6][3] ={{255, 0, 0}, {0,255,0}, {0,0,255}, {255,255,0}, {0,255,255}, {255,0,255}};
  pcl::visualization::CloudViewer viewer("cluster viewer");
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZRGB>);
  // pcl::copyPointCloud(*cloud_filtered, *cloud_cluster);
  pcl::copyPointCloud(*cloud, *cloud_cluster);
  pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
  kdtree.setInputCloud(cloud);
  for (int i(0); i<cloud_cluster->points.size(); ++i)
  {
    cloud_cluster->points[i].r = 128;
    cloud_cluster->points[i].g = 128;
    cloud_cluster->points[i].b = 128;
  }
  for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
  {
    for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); pit++)
    {
      pcl::PointXYZ search_point;
      search_point.x = cloud_filtered->points[*pit].x;
      search_point.y = cloud_filtered->points[*pit].y;
      search_point.z = cloud_filtered->points[*pit].z;
      std::vector<int> k_indices;
      std::vector<float> k_sqr_distances;

      kdtree.radiusSearch(search_point,0.01,k_indices,k_sqr_distances);
      for (std::vector<int>::const_iterator nitr(k_indices.begin()),nlast(k_indices.end()); nitr!=nlast; ++nitr)
      {
        cloud_cluster->points[*nitr].r = colors[j%6][0];
        cloud_cluster->points[*nitr].g = colors[j%6][1];
        cloud_cluster->points[*nitr].b = colors[j%6][2];
      }
    }
    // std::cout << "PointCloud representing the Cluster: " << cloud_cluster->points.size () << " data points." << std::endl;
    // std::stringstream ss;
    // ss << "data/cloud_cluster_" << j << ".pcd";
    // writer.write<pcl::PointXYZRGB> (ss.str (), *cloud_cluster, false); //*
    ++j;
  }

  // Show cloud_cluster only:
  viewer.showCloud (cloud_cluster);

  // Show original cloud + cloud_cluster:
  // pcl::PointCloud<pcl::PointXYZRGB> cloud_c;
  // cloud_c= cloud_org;
  // cloud_c+= *cloud_cluster;
  // viewer.showCloud (cloud_c.makeShared());

  while (!viewer.wasStopped ()) {}

  return (0);
}
