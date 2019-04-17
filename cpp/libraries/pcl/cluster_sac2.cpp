/*
refs:
  http://ros-robot.blogspot.com/2011/08/point-cloud-library_25.html
  http://pointclouds.org/documentation/tutorials/cluster_extraction.php
  http://pointclouds.org/documentation/tutorials/kdtree_search.php
  http://docs.pointclouds.org/trunk/classpcl_1_1_kd_tree.html
  http://pointclouds.org/documentation/tutorials/matrix_transform.php
  http://pointclouds.org/documentation/tutorials/normal_estimation.php
compile:
  x++ cluster_sac2.cpp -- -I/home/akihiko/install/boost1.49/include -I/usr/include/eigen3 -I/opt/ros/groovy/include -I/opt/ros/groovy/include/pcl-1.6 -I/usr/include/vtk-5.8 -L/opt/ros/groovy/lib -lpcl_visualization -lpcl_search -lpcl_kdtree -lpcl_filters -lpcl_features -lpcl_segmentation -lpcl_io -lpcl_common -lrostime -lvtkCommon -lvtkFiltering -lvtkRendering -lvtkGraphics -lboost_system-mt -lboost_thread-mt -Wl,-rpath /opt/ros/groovy/lib
usage:
  Extract clusters, then cylinder model fitting (RANSAC) is applied to each cluster; the program source code is much more readable
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

class TPCLViewer
{
public:
  TPCLViewer()
    {
      Init();
    }

  TPCLViewer(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud, const std::string &name="cloud1", int point_size=1)
    {
      Init();
      AutoCamera(cloud, name, point_size);
    }

  void Init()
    {
      viewer_= boost::shared_ptr<pcl::visualization::PCLVisualizer>(new pcl::visualization::PCLVisualizer ("3D Viewer"));
      viewer_->setBackgroundColor (0, 0, 0);
      viewer_->addCoordinateSystem (0.2);
    }

  bool IsStopped()  {return viewer_->wasStopped();}
  void SpinOnce(int time=1, bool force_redraw=false)
    {
      viewer_->spinOnce(time,force_redraw);
    }

  void AutoCamera(
      const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud,
      const std::string &name="cloud1",
      int point_size=1)
    {
      pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
      viewer_->addPointCloud<pcl::PointXYZRGB> (cloud, rgb, name);
      viewer_->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, point_size, name);

      viewer_->initCameraParameters ();
      viewer_->resetCameraViewpoint(name);
    }

  void AddPointCloud(
      const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud,
      const std::string &name,
      int point_size=1)
    {
      pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
      viewer_->addPointCloud<pcl::PointXYZRGB> (cloud, rgb, name);
      viewer_->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, point_size, name);
    }

  void AddCylinder(
      const pcl::ModelCoefficients::Ptr &coefficients_std,
      const pcl::ModelCoefficients::Ptr &coefficients_ext,
      const std::string &name, int r=255, int g=255, int b=255)
    {
      // viewer_->addCylinder (*coefficients_std, name);
      const std::vector<float> &c_std(coefficients_std->values);
      const std::vector<float> &c_ext(coefficients_ext->values);
      double radius= c_std[6];
      double len= c_ext[3];

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
      Eigen::Vector3d cyl_axis(c_std[3],c_std[4],c_std[5]);
      cyl_axis.normalize();
      Eigen::Vector3d rot_axis= Eigen::Vector3d::UnitZ().cross(cyl_axis);
      rot_axis.normalize();
      double rot_angle= std::acos(Eigen::Vector3d::UnitZ().transpose()*cyl_axis);

      Eigen::Affine3d TA;
      TA= Eigen::Translation3d(c_ext[0],c_ext[1],c_ext[2]) * Eigen::AngleAxisd(rot_angle,rot_axis);
      Eigen::Matrix4f TM(TA.matrix().cast<float>());
      std::cerr<<"Cylinder trans:\n"<<TM<<std::endl;

      // Executing the transformation
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cyl(new pcl::PointCloud<pcl::PointXYZRGB>());
      pcl::transformPointCloud(*cloud_bcyl, *cloud_cyl, TM);

      // AddPointCloud(cloud_bcyl, name+"_b", 1);
      AddPointCloud(cloud_cyl, name, 1);
    }

  void RemoveAll()
    {
      viewer_->removeAllPointClouds();
    }

private:
  boost::shared_ptr<pcl::visualization::PCLVisualizer>  viewer_;

};


// Apply VoxelGrid filter of leaf size (lx,ly,lz)
template <typename t_point>
void DownsampleByVoxelGrid(
    const typename pcl::PointCloud<t_point>::ConstPtr &cloud_in,
    typename pcl::PointCloud<t_point>::Ptr &cloud_out,
    float lx, float ly, float lz)
{
  // Create the filtering object: downsample the dataset
  pcl::VoxelGrid<t_point> vg;
  vg.setInputCloud (cloud_in);
  vg.setLeafSize (lx,ly,lz);
  vg.filter (*cloud_out);
}

// Remove planar objects from a point cloud
template <typename t_point>
bool RemovePlains(
    typename pcl::PointCloud<t_point>::Ptr &cloud_io,
    const double &non_planar_points_ratio,
    const double &ransac_dist_thresh,
    int ransac_max_iterations)
{
  // Create the segmentation object for the planar model and set all the parameters
  pcl::SACSegmentation<t_point>  seg;
  pcl::PointIndices::Ptr  inliers(new pcl::PointIndices);
  pcl::ModelCoefficients::Ptr  coefficients(new pcl::ModelCoefficients);
  // pcl::PointCloud<t_point>::Ptr  cloud_plane(new pcl::PointCloud<t_point> ());
  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setMaxIterations(ransac_max_iterations);
  seg.setDistanceThreshold(ransac_dist_thresh);

  int nr_points = cloud_io->points.size ();
  while (cloud_io->points.size() > non_planar_points_ratio*nr_points)
  {
    // Segment the largest planar component from the remaining cloud
    seg.setInputCloud(cloud_io);
    seg.segment (*inliers, *coefficients); //*
    if (inliers->indices.size () == 0)
    {
      std::cerr<<"Error: Could not estimate a planar model for the given dataset."<<std::endl;
      return false;
    }

    // Extract the planar inliers from the input cloud
    pcl::ExtractIndices<t_point> extract;
    extract.setInputCloud (cloud_io);
    extract.setIndices (inliers);

    // Remove the planar inliers, extract the rest
    extract.setNegative (true);
    extract.filter (*cloud_io);
  }
  return true;
}

// Extract clusters from a point cloud
template <typename t_point>
void ExtractClusters(
    const typename pcl::PointCloud<t_point>::ConstPtr &cloud_in,
    std::vector<pcl::PointIndices> &cluster_indices,
    const double &cluster_tol,
    int min_cluster_size, int max_cluster_size)
{
  // Creating the KdTree object for the search method of the extraction
  typename pcl::search::KdTree<t_point>::Ptr tree (new pcl::search::KdTree<t_point>);
  tree->setInputCloud (cloud_in);

  pcl::EuclideanClusterExtraction<t_point> ec;
  ec.setClusterTolerance(cluster_tol);
  ec.setMinClusterSize(min_cluster_size);
  ec.setMaxClusterSize(max_cluster_size);
  ec.setSearchMethod(tree);
  ec.setInputCloud(cloud_in);
  ec.extract(cluster_indices);
}

static void ColorPointCloud(
    const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &cloud_in,
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud_out,
    int r, int g, int b)
{
  pcl::copyPointCloud(*cloud_in, *cloud_out);
  for (int i(0); i<cloud_out->points.size(); ++i)
  {
    cloud_out->points[i].r = r;
    cloud_out->points[i].g = g;
    cloud_out->points[i].b = b;
  }
}

template <typename t_point>
class TNeighborSearcher
{
public:
  TNeighborSearcher(const typename pcl::PointCloud<t_point>::ConstPtr &cloud_in)
    {
      kdtree_.setInputCloud(cloud_in);
      tmp_indices_.resize(cloud_in->points.size());
    }

  void SearchFromPoints(
      const typename pcl::PointCloud<t_point>::ConstPtr &cloud_ref,
      const pcl::PointIndices &ref_indices,
      pcl::PointIndices::Ptr &indices_out,
      const double &radius)
    {
      std::vector<int> &ext_indices(indices_out->indices);
      ext_indices.clear();
      for(std::vector<int>::const_iterator pit(ref_indices.indices.begin()),plast(ref_indices.indices.end()); pit!=plast; ++pit)
      {
        t_point search_point;
        search_point.x = cloud_ref->points[*pit].x;
        search_point.y = cloud_ref->points[*pit].y;
        search_point.z = cloud_ref->points[*pit].z;
        std::vector<int> k_indices;
        std::vector<float> k_sqr_distances;

        kdtree_.radiusSearch(search_point,radius,k_indices,k_sqr_distances);
        std::sort(k_indices.begin(),k_indices.end());
        std::vector<int>::iterator last_itr= std::set_union(ext_indices.begin(), ext_indices.end(), k_indices.begin(), k_indices.end(), tmp_indices_.begin());
        ext_indices.resize(last_itr-tmp_indices_.begin());
        std::copy(tmp_indices_.begin(),last_itr, ext_indices.begin());
      }
    }

private:
  pcl::KdTreeFLANN<t_point> kdtree_;
  std::vector<int> tmp_indices_;
};

template <typename t_point>
void ExtractByIndices(
    const typename pcl::PointCloud<t_point>::ConstPtr &cloud_in,
    typename pcl::PointCloud<t_point>::Ptr &cloud_out,
    const pcl::PointIndices::ConstPtr &indices)
{
  pcl::ExtractIndices<t_point> extract;
  extract.setInputCloud(cloud_in);
  extract.setIndices(indices);
  extract.setNegative(false);
  extract.filter(*cloud_out);
}

/* Analyze the cylinder information to get the center position and the length.
    coefficients_ext: [0-2]: center x,y,z, [3]: length */
template <typename t_point>
void GetCylinderProp(
    const typename pcl::PointCloud<t_point>::ConstPtr &cloud_obj,
    pcl::PointIndices::ConstPtr inliers_cyl,
    const pcl::ModelCoefficients::Ptr &coefficients_std,
    pcl::ModelCoefficients::Ptr &coefficients_ext)
{
  const std::vector<float> &c_std(coefficients_std->values);
  std::vector<float> &c_ext(coefficients_ext->values);
  Eigen::Vector3d cyl_axis(c_std[3],c_std[4],c_std[5]);
  cyl_axis.normalize();
  Eigen::Vector3d cyl_point(c_std[0],c_std[1],c_std[2]);

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

  // std::cerr<<"tmin: "<<tmin<<std::endl;
  // std::cerr<<"tmax: "<<tmax<<std::endl;
  Eigen::Vector3d cyl_center= cyl_point + 0.5*(tmax+tmin)*cyl_axis;
  c_ext.resize(4);
  c_ext[0]= cyl_center[0];
  c_ext[1]= cyl_center[1];
  c_ext[2]= cyl_center[2];
  c_ext[3]= tmax-tmin;
}

/* Extract a cylinder from a point cloud.
    coefficients_std: [0-2]: point on axis, [3-5]: axis, [6]: radius,
    coefficients_ext: [0-2]: center x,y,z, [3]: length.
  \return The ratio of the number of inliers per the number of cloud_in.
  \note If the ratio of the number of inliers per the number of cloud_in is smaller than cylinder_ratio_thresh, coefficients_ext is not computed. */
template <typename t_point>
double ExtractCylinder(
    const typename pcl::PointCloud<t_point>::ConstPtr &cloud_in,
    pcl::PointIndices::Ptr &inliers,
    pcl::ModelCoefficients::Ptr &coefficients_std,
    pcl::ModelCoefficients::Ptr &coefficients_ext,
    int normal_est_k,
    const double &ransac_normal_dist_w,
    const double &ransac_dist_thresh,
    const double &ransac_radius_min,
    const double &ransac_radius_max,
    int ransac_max_iterations,
    const double &cylinder_ratio_thresh)
{
  // Estimate point normals
  pcl::PointCloud<pcl::Normal>::Ptr  cloud_normals(new pcl::PointCloud<pcl::Normal>);
  pcl::NormalEstimation<t_point, pcl::Normal>  ne;
  typename pcl::search::KdTree<t_point>::Ptr  tree(new pcl::search::KdTree<t_point>());
  ne.setSearchMethod(tree);
  ne.setInputCloud(cloud_in);
  ne.setKSearch(normal_est_k);
  ne.compute(*cloud_normals);

  // Create the segmentation object for cylinder segmentation and set all the parameters
  pcl::SACSegmentationFromNormals<t_point, pcl::Normal>  seg;
  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_CYLINDER);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setNormalDistanceWeight(ransac_normal_dist_w);
  seg.setMaxIterations(ransac_max_iterations);
  seg.setDistanceThreshold(ransac_dist_thresh);
  seg.setRadiusLimits(ransac_radius_min, ransac_radius_max);
  seg.setInputCloud(cloud_in);
  seg.setInputNormals(cloud_normals);
  // Obtain the cylinder inliers and coefficients
  seg.segment (*inliers, *coefficients_std);

  double cratio= (double)inliers->indices.size() / (double)cloud_in->points.size();
  if(cratio<cylinder_ratio_thresh)  return cratio;

  GetCylinderProp<t_point>(cloud_in,inliers,coefficients_std,coefficients_ext);
  return cratio;
}


int main (int argc, char** argv)
{
  // Read in the cloud data
  pcl::PointCloud<pcl::PointXYZRGB> cloud_org;
  if(argc==1)
    pcl::io::loadPCDFile("table_scene_lms400.pcd", cloud_org);
  else
    pcl::io::loadPCDFile(argv[1], cloud_org);

  pcl::PointCloud<pcl::PointXYZ>::Ptr  cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::copyPointCloud(cloud_org,*cloud);
  std::cerr<<"Number of original points: "<<cloud->points.size()<<std::endl;

  // Downsample the dataset using a leaf size of 0.5cm
  pcl::PointCloud<pcl::PointXYZ>::Ptr  cloud_low(new pcl::PointCloud<pcl::PointXYZ>);
  DownsampleByVoxelGrid<pcl::PointXYZ>(cloud, cloud_low, 0.005, 0.005, 0.005);
  std::cerr<<"Number of downsampled points: "<<cloud_low->points.size()<<std::endl;

  // Remove plains
  bool res= RemovePlains<pcl::PointXYZ>(cloud_low,
                /*non_planar_points_ratio=*/0.3,
                /*ransac_dist_thresh=*/0.02,
                /*ransac_max_iterations=*/100);
  if(!res)  return -1;

  // Extract clusters
  std::vector<pcl::PointIndices> cluster_indices;
  ExtractClusters<pcl::PointXYZ>(cloud_low, cluster_indices,
      /*cluster_tol=*/0.02/*2cm*/,
      /*min_cluster_size=*/100,
      /*max_cluster_size=*/25000);
  std::cerr<<"Clusters found: "<<cluster_indices.size()<<std::endl;


  float colors[6][3] ={{255, 0, 0}, {0,255,0}, {0,0,255}, {255,255,0}, {0,255,255}, {255,0,255}};

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr  cloud_col(new pcl::PointCloud<pcl::PointXYZRGB>);
  ColorPointCloud(cloud, cloud_col, 128,128,128);
  // Visualizer
  TPCLViewer viewer;
  viewer.AutoCamera(cloud_col);


  TNeighborSearcher<pcl::PointXYZ>  neighbor_searcher(cloud);
  int i_cyl(0);
  for(size_t j(0); j<cluster_indices.size(); ++j)
  {
    pcl::PointIndices::Ptr ext_indices(new pcl::PointIndices);
    neighbor_searcher.SearchFromPoints(
        cloud_low,
        cluster_indices[j],
        ext_indices,
        /*radius=*/0.01);

    typedef pcl::PointXYZRGB PointT;
    pcl::PointCloud<PointT>::Ptr  cloud_obj(new pcl::PointCloud<PointT>);
    ExtractByIndices<PointT>(cloud_org.makeShared(), cloud_obj, ext_indices);

    // Apply RANSAC for each cluster to get cylinder
    pcl::PointIndices::Ptr inliers_cylinder(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr cyl_coefficients_std(new pcl::ModelCoefficients);
    pcl::ModelCoefficients::Ptr cyl_coefficients_ext(new pcl::ModelCoefficients);
    double cratio, cylinder_ratio_thresh;
    cratio= ExtractCylinder<PointT>(
        cloud_obj,
        inliers_cylinder,
        cyl_coefficients_std,
        cyl_coefficients_ext,
        /*normal_est_k=*/50,
        /*ransac_normal_dist_w=*/0.1,
        /*ransac_dist_thresh=*/0.05,
        /*ransac_radius_min=*/0.0,
        /*ransac_radius_max=*/0.5,
        /*ransac_max_iterations=*/1000,
        cylinder_ratio_thresh=0.2);

    // Show in half of original color:
    for (int i(0); i<cloud_obj->points.size(); ++i)
    {
      cloud_obj->points[i].r /= 2;
      cloud_obj->points[i].g /= 2;
      cloud_obj->points[i].b /= 2;
    }

    std::cerr<<"-----------------------"<<std::endl;
    std::cerr<<"Cluster "<<j<<std::endl;
    std::cerr<<"Cylinder ratio: "<<cratio<<std::endl;
    if(cratio < cylinder_ratio_thresh)
    {
      viewer.AddPointCloud(cloud_obj,GetID("cloud_obj",j),2);
      continue;
    }

    std::cerr<<"Cylinder "<<i_cyl<<std::endl;
    std::cerr<<"Cylinder coefficients_std: "<<*cyl_coefficients_std<<std::endl;
    std::cerr<<"Cylinder coefficients_ext: "<<*cyl_coefficients_ext<<std::endl;

    // Visualize the cylinder inliers
    for (int i(0); i<inliers_cylinder->indices.size(); ++i)
    {
      cloud_obj->points[inliers_cylinder->indices[i]].r = (colors[i_cyl%6][0]+cloud_obj->points[inliers_cylinder->indices[i]].r)/2;
      cloud_obj->points[inliers_cylinder->indices[i]].g = (colors[i_cyl%6][1]+cloud_obj->points[inliers_cylinder->indices[i]].g)/2;
      cloud_obj->points[inliers_cylinder->indices[i]].b = (colors[i_cyl%6][2]+cloud_obj->points[inliers_cylinder->indices[i]].b)/2;
    }

    viewer.AddPointCloud(cloud_obj,GetID("cloud_obj",j),2);
    viewer.AddCylinder(cyl_coefficients_std,cyl_coefficients_ext,GetID("cloud_obj_cyl",j));

    ++i_cyl;
  }

  while(!viewer.IsStopped())  {viewer.SpinOnce(100);}

  return (0);
}
