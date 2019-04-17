/*
refs:
  http://ros-robot.blogspot.com/2011/08/point-cloud-library_25.html
  http://pointclouds.org/documentation/tutorials/cluster_extraction.php
  http://pointclouds.org/documentation/tutorials/kdtree_search.php
  http://docs.pointclouds.org/trunk/classpcl_1_1_kd_tree.html
  http://pointclouds.org/documentation/tutorials/matrix_transform.php
  http://pointclouds.org/documentation/tutorials/normal_estimation.php
compile:
  x++ cluster_sac.cpp -- -I/home/akihiko/install/boost1.49/include -I/usr/include/eigen3 -I/opt/ros/groovy/include -I/opt/ros/groovy/include/pcl-1.6 -I/usr/include/vtk-5.8 -L/opt/ros/groovy/lib -lpcl_visualization -lpcl_search -lpcl_kdtree -lpcl_filters -lpcl_features -lpcl_segmentation -lpcl_io -lpcl_common -lrostime -lvtkCommon -lvtkFiltering -lvtkRendering -lvtkGraphics -lboost_system-mt -lboost_thread-mt -Wl,-rpath /opt/ros/groovy/lib
usage:
  Extract clusters, then cylinder model fitting (RANSAC) is applied to each cluster
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
#include <pcl/common/transforms.h>
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

void AddCylinder(
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer,
    const pcl::ModelCoefficients &coefficients,
    const pcl::ModelCoefficients &coefficients2,
    const std::string &name, int r=255, int g=255, int b=255)
{
  // viewer->addCylinder (coefficients, name);

  double radius= coefficients.values[6];
  double len= coefficients2.values[3];

  if(radius<0.0 || len<0.0)
  {
    std::cerr<<"#Invalid cylinder"<<std::endl;
    return;
  }

  // Generate a basic cylinder
  int num_z(20), num_th(40);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_bcyl(new pcl::PointCloud<pcl::PointXYZRGB>());
  cloud_bcyl->width    = num_z*num_th;
  cloud_bcyl->height   = 1;
  cloud_bcyl->is_dense = false;
  cloud_bcyl->points.resize(cloud_bcyl->width * cloud_bcyl->height);

  size_t i(0);
  for(int iz(0); iz<num_z; ++iz)
  {
    double z= -0.5*len + static_cast<double>(iz)*len/static_cast<double>(num_z-1);
    for(int ith(0); ith<num_th; ++ith)
    {
      double th= static_cast<double>(ith)*2.0*M_PI/static_cast<double>(num_th-1);
      double x= radius*std::cos(th), y= radius*std::sin(th);
      cloud_bcyl->points[i].x= x;
      cloud_bcyl->points[i].y= y;
      cloud_bcyl->points[i].z= z;
      cloud_bcyl->points[i].r= r;
      cloud_bcyl->points[i].g= g;
      cloud_bcyl->points[i].b= b;
      ++i;
    }
  }
  // pcl::io::savePCDFileASCII ("/tmp/test_pcd.pcd", *cloud_bcyl);

  // Rotate and translate the cylinder
  Eigen::Vector3d cyl_axis(coefficients.values[3],coefficients.values[4],coefficients.values[5]);
  cyl_axis.normalize();
  Eigen::Vector3d rot_axis= Eigen::Vector3d::UnitZ().cross(cyl_axis);
  rot_axis.normalize();
  double rot_angle= std::acos(Eigen::Vector3d::UnitZ().transpose()*cyl_axis);

  Eigen::Affine3d TA;
  TA= Eigen::Translation3d(coefficients2.values[0],coefficients2.values[1],coefficients2.values[2]) * Eigen::AngleAxisd(rot_angle,rot_axis);
  Eigen::Matrix4f TM(TA.matrix().cast<float>());
  std::cerr<<"Cylinder trans:\n"<<TM<<std::endl;

  // Executing the transformation
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cyl(new pcl::PointCloud<pcl::PointXYZRGB>());
  pcl::transformPointCloud(*cloud_bcyl, *cloud_cyl, TM);

  // AddCloud(viewer, cloud_bcyl, name+"_b", 1);
  AddCloud(viewer, cloud_cyl, name, 1);
}

/* Analyze the cylinder information, and get the center position and the length.
    ext_coefficients: [0-2]: center x,y,z, [3]: length */
void GetCylinderProp(
    pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud_obj,
    pcl::PointIndices::ConstPtr inliers_cyl,
    const pcl::ModelCoefficients &coefficients,
    pcl::ModelCoefficients &ext_coefficients)
{
  #if 0
  // Extract point cloud of the cylinder
  pcl::PointCloud<PointXYZRGB>::Ptr cloud_cyl(new pcl::PointCloud<PointXYZRGB>);
  pcl::ExtractIndices<PointXYZRGB> extract;
  extract.setInputCloud (cloud_obj);
  extract.setIndices (inliers_cyl);
  extract.setNegative (false);
  extract.filter (*cloud_cyl);

  // Rotate and translate the cylinder
  Eigen::Vector3d cyl_axis(coefficients.values[3],coefficients.values[4],coefficients.values[5]);
  cyl_axis.normalize();
  Eigen::Vector3d rot_axis= cyl_axis.cross(Eigen::Vector3d::UnitZ());
  rot_axis.normalize();
  double rot_angle= std::acos(cyl_axis.transpose()*Eigen::Vector3d::UnitZ());

  Eigen::Affine3d TA;
  TA= Eigen::AngleAxisd(rot_angle,rot_axis) * Eigen::Translation3d(-coefficients.values[0],-coefficients.values[1],-coefficients.values[2]);
  Eigen::Matrix4f TM(TA.matrix().cast<float>());

  // Executing the transformation
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_bcyl(new pcl::PointCloud<pcl::PointXYZRGB>());
  pcl::transformPointCloud(*cloud_cyl, *cloud_bcyl, TM);
  #endif

  Eigen::Vector3d cyl_axis(coefficients.values[3],coefficients.values[4],coefficients.values[5]);
  cyl_axis.normalize();
  Eigen::Vector3d cyl_point(coefficients.values[0],coefficients.values[1],coefficients.values[2]);

  // Project each point on the cylinder axis, compute their minimum and maximum
  double tmin(1.0e10), tmax(-1.0e10);
  for (int i(0); i<inliers_cyl->indices.size(); ++i)
  {
    // Project the point on the cylinder axis
    Eigen::Vector3d p(cloud_obj->points[inliers_cyl->indices[i]].x,cloud_obj->points[inliers_cyl->indices[i]].y,cloud_obj->points[inliers_cyl->indices[i]].z);
    double t= (p-cyl_point).transpose()*cyl_axis;
    if(t<tmin)  tmin= t;
    if(t>tmax)  tmax= t;
  }

  std::cerr<<"tmin: "<<tmin<<std::endl;
  std::cerr<<"tmax: "<<tmax<<std::endl;
  Eigen::Vector3d cyl_center= cyl_point + 0.5*(tmax+tmin)*cyl_axis;
  ext_coefficients.values.resize(4);
  ext_coefficients.values[0]= cyl_center[0];
  ext_coefficients.values[1]= cyl_center[1];
  ext_coefficients.values[2]= cyl_center[2];
  ext_coefficients.values[3]= tmax-tmin;
}

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

  // Create the filtering object: downsample the dataset using a leaf size of 0.5cm
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
  // pcl::visualization::CloudViewer viewer("cluster viewer");
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

  // Visualize
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
  viewer= SetupViewer(cloud_cluster);
  // viewer= SetupViewer(cloud_org.makeShared());

  std::vector<int> tmp_indices(cloud->points.size());
  for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
  {
    pcl::PointIndices::Ptr ext_indices_pcl(new pcl::PointIndices);
    std::vector<int> &ext_indices(ext_indices_pcl->indices);
    for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); pit++)
    {
      pcl::PointXYZ search_point;
      search_point.x = cloud_filtered->points[*pit].x;
      search_point.y = cloud_filtered->points[*pit].y;
      search_point.z = cloud_filtered->points[*pit].z;
      std::vector<int> k_indices;
      std::vector<float> k_sqr_distances;

      kdtree.radiusSearch(search_point,0.01,k_indices,k_sqr_distances);
      std::sort(k_indices.begin(),k_indices.end());
      std::vector<int>::iterator last_itr= std::set_union(ext_indices.begin(), ext_indices.end(), k_indices.begin(), k_indices.end(), tmp_indices.begin());
      ext_indices.resize(last_itr-tmp_indices.begin());
      std::copy(tmp_indices.begin(),last_itr, ext_indices.begin());
    }
    // for (std::vector<int>::const_iterator nitr(ext_indices.begin()),nlast(ext_indices.end()); nitr!=nlast; ++nitr)
    // {
      // cloud_cluster->points[*nitr].r = colors[j%6][0];
      // cloud_cluster->points[*nitr].g = colors[j%6][1];
      // cloud_cluster->points[*nitr].b = colors[j%6][2];
    // }



    // Apply RANSAC for each cluster to get cylinder
    /*ext_indices_pcl*/{
      typedef pcl::PointXYZRGB PointT;

      // Extract point cloud of an object
      pcl::PointCloud<PointT>::Ptr cloud_obj(new pcl::PointCloud<PointT>);
      pcl::ExtractIndices<PointT> extract;
      // extract.setInputCloud (cloud_cluster);
      extract.setInputCloud (cloud_org.makeShared());
      extract.setIndices (ext_indices_pcl);
      extract.setNegative (false);
      extract.filter (*cloud_obj);

      // for (int i(0); i<cloud_obj->points.size(); ++i)
      // {
        // cloud_obj->points[i].r = 255;
        // cloud_obj->points[i].g = 255;
        // cloud_obj->points[i].b = 255;
      // }
      // Show in half of original color:
      for (int i(0); i<cloud_obj->points.size(); ++i)
      {
        cloud_obj->points[i].r /= 2;
        cloud_obj->points[i].g /= 2;
        cloud_obj->points[i].b /= 2;
      }

      // Estimate point normals
      pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
      pcl::NormalEstimation<PointT, pcl::Normal> ne;
      pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT> ());
      ne.setSearchMethod (tree);
      ne.setInputCloud (cloud_obj);
      ne.setKSearch (50);
      ne.compute (*cloud_normals);

      // Create the segmentation object for cylinder segmentation and set all the parameters
      pcl::SACSegmentationFromNormals<PointT, pcl::Normal> seg;
      pcl::ModelCoefficients::Ptr coefficients_cylinder(new pcl::ModelCoefficients);
      pcl::PointIndices::Ptr inliers_cylinder(new pcl::PointIndices);
      seg.setOptimizeCoefficients (true);
      seg.setModelType (pcl::SACMODEL_CYLINDER);
      seg.setMethodType (pcl::SAC_RANSAC);
      seg.setNormalDistanceWeight (0.1);
      seg.setMaxIterations (10000);
      seg.setDistanceThreshold (0.05);
      seg.setRadiusLimits (0, 0.5);
      seg.setInputCloud (cloud_obj);
      seg.setInputNormals (cloud_normals);
      // Obtain the cylinder inliers and coefficients
      seg.segment (*inliers_cylinder, *coefficients_cylinder);
      std::cerr << "Cylinder coefficients: " << *coefficients_cylinder << std::endl;

      // Visualize the cylinder inliers
      for (int i(0); i<inliers_cylinder->indices.size(); ++i)
      {
        cloud_obj->points[inliers_cylinder->indices[i]].r = (colors[j%6][0]+cloud_obj->points[inliers_cylinder->indices[i]].r)/2;
        cloud_obj->points[inliers_cylinder->indices[i]].g = (colors[j%6][1]+cloud_obj->points[inliers_cylinder->indices[i]].g)/2;
        cloud_obj->points[inliers_cylinder->indices[i]].b = (colors[j%6][2]+cloud_obj->points[inliers_cylinder->indices[i]].b)/2;
      }

      *cloud_cluster+= *cloud_obj;
      AddCloud(viewer,cloud_obj,GetID("cloud_obj",j),2);

      pcl::ModelCoefficients::Ptr coefficients_cylinder2(new pcl::ModelCoefficients);
      GetCylinderProp(cloud_obj,inliers_cylinder,*coefficients_cylinder,*coefficients_cylinder2);
      std::cerr << "Cylinder coefficients2: " << *coefficients_cylinder2 << std::endl;

      AddCylinder(viewer,*coefficients_cylinder,*coefficients_cylinder2,GetID("cloud_obj_cyl",j));
    }


    // std::cout << "PointCloud representing the Cluster: " << cloud_cluster->points.size () << " data points." << std::endl;
    // std::stringstream ss;
    // ss << "data/cloud_cluster_" << j << ".pcd";
    // writer.write<pcl::PointXYZRGB> (ss.str (), *cloud_cluster, false); //*
    ++j;
  }

  // Show cloud_cluster only:
  // viewer.showCloud (cloud_cluster);

  // Show original cloud + cloud_cluster:
  // pcl::PointCloud<pcl::PointXYZRGB> cloud_c;
  // cloud_c= cloud_org;
  // cloud_c+= *cloud_cluster;
  // viewer.showCloud (cloud_c.makeShared());

  while(!viewer->wasStopped())  {viewer->spinOnce(100);}

  return (0);
}
