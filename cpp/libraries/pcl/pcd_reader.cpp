/*
src:
  -
compile:
  x++ pcd_reader.cpp -- -I/home/akihiko/install/boost1.49/include -I/usr/include/eigen3 -I/opt/ros/groovy/include -I/opt/ros/groovy/include/pcl-1.6 -I/usr/include/vtk-5.8 -L/opt/ros/groovy/lib -lpcl_visualization -lpcl_io -lpcl_common -Wl,-rpath /opt/ros/groovy/lib
usage:
  Visualize a given point cloud data
  ./a.out
  ./a.out POINT_CLOUD.pcd
*/
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>

int main (int argc, char** argv)
{
  // Read in the cloud data
  pcl::PointCloud<pcl::PointXYZRGB> cloud;
  if(argc==1)
    pcl::io::loadPCDFile("table_scene_lms400.pcd", cloud);
  else
    pcl::io::loadPCDFile(argv[1], cloud);


  std::cout << "PointCloud has: " << cloud.points.size () << " data points." << std::endl;
  double min[]={1e10,1e10,1e10}, max[]={-1e10,-1e10,-1e10};
  for(size_t i(0); i < cloud.points.size (); ++i)
  {
    if(cloud.points[i].x<min[0])  min[0]= cloud.points[i].x;
    if(cloud.points[i].y<min[1])  min[1]= cloud.points[i].y;
    if(cloud.points[i].z<min[2])  min[2]= cloud.points[i].z;
    if(cloud.points[i].x>max[0])  max[0]= cloud.points[i].x;
    if(cloud.points[i].y>max[1])  max[1]= cloud.points[i].y;
    if(cloud.points[i].z>max[2])  max[2]= cloud.points[i].z;
  }
  std::cout<<"X-range: "<<min[0]<<", "<<max[0]<<std::endl;
  std::cout<<"Y-range: "<<min[1]<<", "<<max[1]<<std::endl;
  std::cout<<"Z-range: "<<min[2]<<", "<<max[2]<<std::endl;

  pcl::visualization::CloudViewer viewer("Simple Cloud Viewer");
  viewer.showCloud(cloud.makeShared());
  while (!viewer.wasStopped ()) {}

  return (0);
}
