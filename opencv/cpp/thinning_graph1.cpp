//-------------------------------------------------------------------------------------------
/*! \file    thinning_graph1.cpp
    \brief   Construct a topological graph from thinning image.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Jul.16, 2021

g++ -g -Wall -O2 -o thinning_graph1.out thinning_graph1.cpp -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lopencv_videoio -I/usr/include/opencv4

$ ./thinning_graph1.out sample/binary1.png
$ ./thinning_graph1.out sample/opencv-logo.png
$ ./thinning_graph1.out sample/water_coins.jpg

*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "thinning/thinning.hpp"
#include "thinning/thinning.cpp"
#include <vector>
#include <cstdio>
#include <sys/time.h>  // gettimeofday
#ifdef LIBRARY
#  include "floyd_apsp.cpp"
#  include "float_trackbar.cpp"
#else
#  define LIBRARY
#  include "floyd_apsp.cpp"
#  include "float_trackbar.cpp"
#  undef LIBRARY
#endif
//-------------------------------------------------------------------------------------------
namespace loco_rabbits
{
}
//-------------------------------------------------------------------------------------------
// using namespace std;
using namespace loco_rabbits;
//-------------------------------------------------------------------------------------------
// #define print(var) PrintContainer((var), #var"= ")
// #define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

inline cv::Point DPToOffset(int dp)
{
  switch(dp)
  {
    case 0:  return cv::Point( 0,  0);
    case 1:  return cv::Point( 0, -1);
    case 2:  return cv::Point(+1, -1);
    case 3:  return cv::Point(+1,  0);
    case 4:  return cv::Point(+1, +1);
    case 5:  return cv::Point( 0, +1);
    case 6:  return cv::Point(-1, +1);
    case 7:  return cv::Point(-1,  0);
    case 8:  return cv::Point(-1, -1);
  }
  return cv::Point();
}
//-------------------------------------------------------------------------------------------

inline int OppositeDP(int dp)
{
  switch(dp)
  {
    case 0:  return 0;
    case 1:  return 5;
    case 2:  return 6;
    case 3:  return 7;
    case 4:  return 8;
    case 5:  return 1;
    case 6:  return 2;
    case 7:  return 3;
    case 8:  return 4;
  }
  return -1;
}
//-------------------------------------------------------------------------------------------

int CircularDP(int dp)  // Circular DP indexing where we consider 1,2,...,7,8,1,2,...
{
  if(dp==0)  return 8;
  if(dp==9)  return 1;
  return dp;
}
//-------------------------------------------------------------------------------------------

enum TPointType
{
  ptNone=0,
  ptIsolated,
  ptTerminal,
  ptIntersection,
  ptMiddleLine
};
//-------------------------------------------------------------------------------------------

struct TAllNeighbors
{
  bool Neighbors[9];  // Neighbors[0] is not used.
  bool& operator[](int dp)  // return the value at dp with considering circular indexing.
    {
      return Neighbors[CircularDP(dp)];
    }
  const bool& operator[](int dp) const  // return the value at dp with considering circular indexing.
    {
      return Neighbors[CircularDP(dp)];
    }
};
//-------------------------------------------------------------------------------------------

void RemoveNonTrueNeighbors(TAllNeighbors &neighbors)
{
  for(int dp(1); dp<9; dp+=2)  // 1,3,5,7: dp at + (plus) neighbors.
  {
    if(neighbors[dp])  // If there is a point, remove the next and previous neighbors.
    {
      neighbors[dp+1]= false;
      neighbors[dp-1]= false;
    }
  }
}
//-------------------------------------------------------------------------------------------

inline int CountNonZero(const TAllNeighbors &neighbors)
{
  int nz(0);
  for(int dp(1); dp<9; ++dp)
    if(neighbors[dp])  ++nz;
  return nz;
}
//-------------------------------------------------------------------------------------------

// // This implementation assumes RemoveNonTrueNeighbors is not applied.
// TPointType CategorizeByNeighbors(const TAllNeighbors &neighbors)
// {
//   int nz= CountNonZero(neighbors);
//   if(nz==0)  return ptIsolated;
//   if(nz==1)  return ptTerminal;
//   if(nz==2)  return ptMiddleLine;
//   if(nz==3)
//   {
//     int n_iso(0);
//     for(int dp(1); dp<9; ++dp)
//       if(neighbors[dp] && (!neighbors[dp-1] && !neighbors[dp+1]))  ++n_iso;
//     if(n_iso<2)  return ptMiddleLine;
//     return ptIntersection;
//   }
//   if(nz>=3)
//   {
//     int n_gap(0);
//     for(int dp(1); dp<9; ++dp)
//       if(!neighbors[dp] && (neighbors[dp-1] && neighbors[dp+1]))  ++n_gap;
//     if(n_gap>=2)  return ptIntersection;
//   }
//   return ptMiddleLine;
// }
// This implementation assumes neighbors is pre-processed with RemoveNonTrueNeighbors.
TPointType CategorizeByNeighbors(const TAllNeighbors &neighbors)
{
  int nz= CountNonZero(neighbors);
  if(nz==0)  return ptIsolated;
  if(nz==1)  return ptTerminal;
  if(nz==2)  return ptMiddleLine;
  return ptIntersection;
}
//-------------------------------------------------------------------------------------------

struct TThinningGraph
{
  struct TNeighbor
  {
    int DP;    // Neighbor point (1-8).
    int Next;  // Neighbor node index.
    int PathLen;  // Path length to Next.
    TNeighbor() : DP(0), Next(-1), PathLen(-1)  {}
    TNeighbor(int dp) : DP(dp), Next(-1), PathLen(-1)  {}
  };
  struct TNode
  {
    cv::Point P;
    std::vector<TNeighbor> Neighbors;
    TNode()  {}
    TNode(const cv::Point &px) : P(px)  {}
  };
  std::vector<TNode> Nodes;
};
//-------------------------------------------------------------------------------------------

// // Test if neighbors[i] is an isolated point (i.e. previous and next point are blank).
// inline bool IsIsolated(const std::vector<TThinningGraph::TNeighbor> &neighbors, int i)
// {
//   if(i>=neighbors.size() or i<0)  return false;
//   if(neighbors.size()==1)  return true;
//   int dp= neighbors[i].DP;
//   if(neighbors.size()==2)
//   {
//     int dp_other= neighbors[(i+1)%2].DP;
//     if(dp>dp_other)  std::swap(dp,dp_other);
//     if(dp==dp_other-1 || (dp==1 && dp_other==8))  return false;
//     return true;
//   }
//   int dp_next= neighbors[(i+1)%neighbors.size()].DP;
//   int dp_prev= neighbors[i>0?(i-1):(neighbors.size()-1)].DP;
//   if(dp==dp_next-1 || (dp==8 && dp_next==1))  return false;
//   if(dp==dp_prev+1 || (dp==1 && dp_prev==8))  return false;
//   return true;
// }
// //-------------------------------------------------------------------------------------------

template<typename t_element>
inline t_element ExtendedAt(const cv::Mat &m, const cv::Point &p)
{
  if(p.x<0 || p.y<0 || p.x>=m.cols || p.y>=m.rows)  return t_element(0);
  return m.at<t_element>(p);
}
//-------------------------------------------------------------------------------------------

// Detect graph nodes (terminal and intersection points) on a thinning image;
// Point types of each pixel are saved into point_types.
TThinningGraph DetectNodesFromThinningImg(const cv::Mat &thinning_img, cv::Mat &point_types)
{
  TThinningGraph graph;
  point_types.create(thinning_img.size(), CV_32S);
  for(int y(0); y<thinning_img.rows; ++y)
  {
    for(int x(0); x<thinning_img.cols; ++x)
    {
      cv::Point px(x,y);
      if(thinning_img.at<uchar>(px)==0)
      {
        point_types.at<int>(px)= int(ptNone);
        continue;
      }
//       TThinningGraph::TNode node(px);
      TAllNeighbors all_neighbors;
      for(int dp(1); dp<9; ++dp)
//       {
//         all_neighbors[dp]= thinning_img.at<uchar>(px+DPToOffset(dp))!=0;
        all_neighbors[dp]= ExtendedAt<uchar>(thinning_img,px+DPToOffset(dp))!=0;
//         if(all_neighbors[dp])
//           node.Neighbors.push_back(TThinningGraph::TNeighbor(dp));
//       }
      // Categorize the point by neighbors.
//       if(node.Neighbors.size()==2)  continue;  // Middle point of a line.
//       // if(node.Neighbors.size()==1): Terminal point.
//       else if(node.Neighbors.size()==3)
//       {
//         int n(0);
//         for(int i(0),i_end(node.Neighbors.size()); i<i_end; ++i)
//           if(IsIsolated(node.Neighbors[i],i))  ++n;
//         if(n<2)  continue;  // Middle point of a line.
//         // Otherwise, it is a crossing point.
//       }
//       else if(node.Neighbors.size()==3)

      RemoveNonTrueNeighbors(all_neighbors);
      TPointType type= CategorizeByNeighbors(all_neighbors);
      point_types.at<int>(px)= int(type);
      if(type==ptTerminal or type==ptIntersection)
      {
// std::cerr<<"  "<<px/*int(thinning_img.at<uchar>(px))*/<<"/"<<CountNonZero(all_neighbors);
        TThinningGraph::TNode node(px);
        for(int dp(1); dp<9; ++dp)
          if(all_neighbors[dp])
            node.Neighbors.push_back(TThinningGraph::TNeighbor(dp));
        graph.Nodes.push_back(node);
        point_types.at<int>(px)+= (graph.Nodes.size()-1)*10;
      }
    }
  }
// std::cerr<<std::endl;
  return graph;
}
//-------------------------------------------------------------------------------------------

// Follow the path starting at nitr->P to neitr->DP.
// point_types: Thinning image where each point is categorized.
// return the node index at the reached point, DP from the point, and path length;
// If a pointer path_points is given, points on the path are stored into it, except for the start/end points.
void FollowPathOnImg(const cv::Point p, int dp, const cv::Mat &point_types,
    int &next_node, int &dp_from_next, int &path_len,
    std::vector<cv::Point> *path_points=NULL)
{
  path_len= 0;
  int dp_prev(dp);
  cv::Point px(p);

  {
    int type_nodeidx= ExtendedAt<int>(point_types,px+DPToOffset(dp));
    TPointType type= TPointType(type_nodeidx%10);
    if(type==ptTerminal || type==ptIntersection)
    {
      next_node= type_nodeidx/10;
      dp_from_next= OppositeDP(dp);
      ++path_len;
      return;
    }
  }

  int DP_ORDER[]= {0, 1,3,5,7, 2,4,6,8};  // + (plus) neighbors -> x (corner) neighbors order.
  while(true)
  {
// std::cerr<<" "<<px<<" "<<ExtendedAt<int>(point_types,px)<<" "<<dp_prev<<" ("<<OppositeDP(dp_prev)<<")"<<std::endl;
    px+= DPToOffset(dp_prev);
    ++path_len;
    if(path_points!=NULL)  path_points->push_back(px);
    dp_prev= OppositeDP(dp_prev);
    int dp, type_nodeidx;
    TPointType type;
    // for(dp=1; dp<9; ++dp)  // 1,3,5,7: dp at + (plus) neighbors.
    bool found(false);
    for(int dpo=1; dpo<9; ++dpo)
    {
      dp= DP_ORDER[dpo];
      if(dp==dp_prev || dp==CircularDP(dp_prev+1) || dp==CircularDP(dp_prev-1))  continue;
      type_nodeidx= ExtendedAt<int>(point_types,px+DPToOffset(dp));
      type= TPointType(type_nodeidx%10);
      if(type==ptTerminal || type==ptIntersection)
      {
        next_node= type_nodeidx/10;
        dp_from_next= OppositeDP(dp);
        ++path_len;
        return;
      }
      if(type==ptMiddleLine)
      {
        found= true;
        break;
      }
    }
// //     bool found(false);
// //     for(dp=1; dp<9; dp+=2)  // 1,3,5,7: dp at + (plus) neighbors.
// //     {
// //       if(dp==dp_prev)  continue;
// //       if(TPointType(ExtendedAt<int>(point_types,px+DPToOffset(dp))%10)==ptMiddleLine)
// //       {
// //         found= true;
// //         break;
// //       }
// //     }
// //     if(!found)
// //     {
// //       for(dp=2; dp<9; dp+=2)  // 2,4,6,8: dp at x (corner) neighbors.
// //       {
// //         if(dp==dp_prev)  continue;
// //         if(TPointType(ExtendedAt<int>(point_types,px+DPToOffset(dp))%10)==ptMiddleLine)
// //         {
// //           found= true;
// //           break;
// //         }
// //       }
// //     }
//     bool found(false);
//     for(dp=1; dp<9; ++dp)
//     {
//       if(dp==dp_prev || dp==CircularDP(dp_prev+1) || dp==CircularDP(dp_prev-1))  continue;
//       if(TPointType(ExtendedAt<int>(point_types,px+DPToOffset(dp))%10)==ptMiddleLine)
//       {
//         found= true;
//         break;
//       }
//     }
    if(!found)  throw;
// if(dp==9)  {std::cerr<<" all_neighbors:";for(int dp(1); dp<9; ++dp)std::cerr<<" "<<(ExtendedAt<int>(point_types,px+DPToOffset(dp))%10);std::cerr<<std::endl;}
    dp_prev= dp;
  }
}
//-------------------------------------------------------------------------------------------

// Connect nodes in graph where the node path lengths are also filled.
void ConnectNodes(TThinningGraph &graph, const cv::Mat &point_types)
{
  typedef TThinningGraph::TNode TNode;
  typedef TThinningGraph::TNeighbor TNeighbor;
  int i_node(0);
  for(std::vector<TNode>::iterator nitr(graph.Nodes.begin()),nitr_end(graph.Nodes.end()); nitr!=nitr_end; ++nitr,++i_node)
  {
    for(std::vector<TNeighbor>::iterator neitr(nitr->Neighbors.begin()),neitr_end(nitr->Neighbors.end()); neitr!=neitr_end; ++neitr)
    {
      if(neitr->PathLen>=0)  continue;
      // Follow the path starting at nitr->P to neitr->DP.
// std::cerr<<"path following: "<<nitr->P<<", "<<neitr->DP<<std::endl;
      int i_next(-1), dp_from_next(-1), path_len(-1);
      FollowPathOnImg(nitr->P, neitr->DP, point_types, i_next, dp_from_next, path_len);
// std::cerr<<"path followed: "<<nitr->P<<", "<<neitr->DP<<" --> "<<graph.Nodes[i_next].P<<", "<<dp_from_next<<" / "<<path_len<<std::endl;
      neitr->Next= i_next;
      neitr->PathLen= path_len;
      bool opposite_assigned(false);
      for(std::vector<TNeighbor>::iterator ne2itr(graph.Nodes[i_next].Neighbors.begin()),ne2itr_end(graph.Nodes[i_next].Neighbors.end()); ne2itr!=ne2itr_end; ++ne2itr)
        if(ne2itr->DP==dp_from_next)
        {
          ne2itr->Next= i_node;
          ne2itr->PathLen= path_len;
          opposite_assigned= true;
        }
      if(!opposite_assigned) std::cerr<<"###Failed opposite assignment: "<<i_node<<"-"<<neitr->DP<<"-->"<<dp_from_next<<"-"<<i_next<<std::endl;
    }
  }
}
//-------------------------------------------------------------------------------------------

// Analyze a thinning graph to get spine features (longest path per graph).
// approx_epsilon: Polygon approximation epsilon (0: do nothing).
void ThinningGraphToSpinePolys(const TThinningGraph &graph, const cv::Mat &point_types, std::vector<std::vector<cv::Point> > &spine_polys, const double &approx_epsilon=3.0)
{
  typedef TThinningGraph::TNode TNode;
  typedef TThinningGraph::TNeighbor TNeighbor;
  int N= graph.Nodes.size();  // Number of nodes.
  std::list<TFloydAPSPEdge>  edges;  // List of edge and cost values: (node1,node2,cost).
  int n(0);
  for(std::vector<TNode>::const_iterator nitr(graph.Nodes.begin()),nitr_end(graph.Nodes.end()); nitr!=nitr_end; ++nitr,++n)
    for(std::vector<TNeighbor>::const_iterator neitr(nitr->Neighbors.begin()),neitr_end(nitr->Neighbors.end()); neitr!=neitr_end; ++neitr)
      edges.push_back(TFloydAPSPEdge(n,neitr->Next,neitr->PathLen));

  //D: matrix containing lowest cost.
  //P: matrix containing a via point on the shortest path.
  cv::Mat D,P;
  FloydAPSP(N, edges, D, P);
  // std::cerr<<"Lowest cost matrix:"<<std::endl<<D<<std::endl;
  // std::cerr<<"Via points on the shortest path matrix:"<<std::endl<<P<<std::endl;
  std::vector<std::vector<int> > iso_graphs= IsolatedGraphs(P);
  std::cerr<<"Isolated graphs: ";
  for(int i(0),i_end(iso_graphs.size());i<i_end;++i)
  {
    std::cerr<<"[";
    for(int j(0),j_end(iso_graphs[i].size());j<j_end;++j)
      std::cerr<<" "<<iso_graphs[i][j];
    std::cerr<<"]";
  }
  std::cout<<std::endl;

  spine_polys.clear();
  for(int i(0),i_end(iso_graphs.size());i<i_end;++i)
  {
    int n1_max(-1),n2_max(-1);
    double path_len_max(-1);
    for(int n1i(0),n1i_end(iso_graphs[i].size());n1i<n1i_end;++n1i)
      for(int n2i(0),n2i_end(iso_graphs[i].size());n2i<n2i_end;++n2i)
      {
        int n1(iso_graphs[i][n1i]), n2(iso_graphs[i][n2i]);
        if(D.at<double>(n1,n2)>path_len_max)
        {
          n1_max= n1;
          n2_max= n2;
          path_len_max= D.at<double>(n1,n2);
        }
      }
    std::list<int> path;
    bool res= ShortestPath(n1_max, n2_max, P, path);
    std::cerr<<"Longest path in graph "<<i<<": ";
    {for(std::list<int>::const_iterator pitr(path.begin()), pitr_end(path.end()); pitr!=pitr_end; ++pitr) std::cerr<<" "<<*pitr;} std::cerr<<std::endl;
    if(!res || path.size()<=1)  continue;
    std::vector<cv::Point> path_points;
    std::list<int>::const_iterator pitr(path.begin()), pitr_end(path.end());
    int n_prev(*pitr);
    for(++pitr; pitr!=pitr_end; ++pitr)
    {
// std::cerr<<"n_prev "<<n_prev<<" / *pitr "<<*pitr<<std::endl;
      path_points.push_back(graph.Nodes[n_prev].P);
      int nei(0), nei_end(graph.Nodes[n_prev].Neighbors.size());
      for(nei=0; nei<nei_end; ++nei)
      {
// std::cerr<<"  n_prev/next: "<<graph.Nodes[n_prev].Neighbors[nei].Next<<"("<<nei<<"/"<<nei_end<<")"<<std::endl;
        if(graph.Nodes[n_prev].Neighbors[nei].Next==*pitr)  break;
      }
      if(nei==nei_end)  throw;
      int i_next(-1), dp_from_next(-1), path_len(-1);
      FollowPathOnImg(graph.Nodes[n_prev].P, graph.Nodes[n_prev].Neighbors[nei].DP, point_types, i_next, dp_from_next, path_len, &path_points);
      if(i_next!=*pitr)  throw;
      n_prev= *pitr;
    }
    path_points.push_back(graph.Nodes[path.back()].P);
    std::vector<cv::Point> spine_poly;
    int poly_step(1);
    for(int p(0),p_end(path_points.size()); p<p_end; p+= poly_step)
      spine_poly.push_back(path_points[p]);
    if(path_points.size()%poly_step!=0)  spine_poly.push_back(path_points.back());
    if(approx_epsilon>0.0)
      cv::approxPolyDP(spine_poly, spine_poly, /*epsilon=*/approx_epsilon, /*closed=*/false);
    if(spine_poly.size()>1)
      spine_polys.push_back(spine_poly);
  }
}
//-------------------------------------------------------------------------------------------

void DrawThinningGraph(cv::Mat &img, const TThinningGraph &graph, const std::vector<std::vector<cv::Point> > &spine_polys, const double &resize_factor=1.0)
{
  typedef TThinningGraph::TNode TNode;
  typedef TThinningGraph::TNeighbor TNeighbor;
  const double &f(resize_factor);
  for(std::vector<TNode>::const_iterator nitr(graph.Nodes.begin()),nitr_end(graph.Nodes.end()); nitr!=nitr_end; ++nitr)
  {
    for(std::vector<TNeighbor>::const_iterator neitr(nitr->Neighbors.begin()),neitr_end(nitr->Neighbors.end()); neitr!=neitr_end; ++neitr)
      cv::line(img, f*nitr->P, f*graph.Nodes[neitr->Next].P, cv::Scalar(255,255,0), 1, 8, 0);
    if(nitr->Neighbors.size()==1)
      cv::circle(img, f*nitr->P, 3, cv::Scalar(0,255,0));
    else if(nitr->Neighbors.size()==3)
      cv::circle(img, f*nitr->P, 3, cv::Scalar(0,0,255));
    else
      cv::circle(img, f*nitr->P, 3, cv::Scalar(255,0,0));
  }
  std::vector<std::vector<cv::Point> > spine_polys_resized;
  for(std::vector<std::vector<cv::Point> >::const_iterator i1(spine_polys.begin()),i1_e(spine_polys.end()); i1!=i1_e; ++i1)
  {
    spine_polys_resized.push_back(std::vector<cv::Point>());
    for(std::vector<cv::Point>::const_iterator i2(i1->begin()),i2_e(i1->end()); i2!=i2_e; ++i2)
      spine_polys_resized.back().push_back(f*(*i2));
  }
  cv::polylines(img, spine_polys_resized, /*isClosed=*/false, cv::Scalar(0,255,255), 2);
}
//-------------------------------------------------------------------------------------------


#ifndef LIBRARY

inline double GetCurrentTime(void)
{
  struct timeval time;
  gettimeofday (&time, NULL);
  return static_cast<double>(time.tv_sec) + static_cast<double>(time.tv_usec)*1.0e-6;
}
//-------------------------------------------------------------------------------------------

int main(int argc, char **argv)
{
  cv::Mat img_in= cv::imread(argv[1], cv::IMREAD_COLOR);

  float resize_factor(0.5);
  bool inv_img(false);
  cv::namedWindow("Input",1);
  CreateTrackbar<bool>("Invert", "Input", &inv_img, &TrackbarPrintOnTrack);
  CreateTrackbar<float>("resize_factor", "Input", &resize_factor, 0.1, 1.0, 0.1, &TrackbarPrintOnTrack);

  while(true)
  {
    cv::Mat img;
    if(resize_factor!=1.0)
      cv::resize(img_in, img, cv::Size(), resize_factor, resize_factor, cv::INTER_LINEAR);
    else
      img= img_in;

    // Threshold the input image
    cv::Mat img_grayscale, img_binary;
    cv::cvtColor(img, img_grayscale, cv::COLOR_BGR2GRAY);
    if(inv_img)
      cv::threshold(img_grayscale, img_binary, 0, 255, cv::THRESH_OTSU | cv::THRESH_BINARY_INV);
    else
      cv::threshold(img_grayscale, img_binary, 0, 255, cv::THRESH_OTSU | cv::THRESH_BINARY);

    // Apply thinning to get a skeleton
    cv::Mat img_thinning_ZS, img_thinning_GH;
    double t0= GetCurrentTime();
    cv::ximgproc::thinning(img_binary, img_thinning_ZS, cv::ximgproc::THINNING_ZHANGSUEN);
    double t1= GetCurrentTime();
    cv::ximgproc::thinning(img_binary, img_thinning_GH, cv::ximgproc::THINNING_GUOHALL);
    double t2= GetCurrentTime();

    // Apply the graph point detection and connection.
    cv::Mat point_types_ZS, point_types_GH;
    std::vector<std::vector<cv::Point> > spine_polys_ZS, spine_polys_GH;
    TThinningGraph graph_ZS= DetectNodesFromThinningImg(img_thinning_ZS, point_types_ZS);
    ConnectNodes(graph_ZS, point_types_ZS);
    ThinningGraphToSpinePolys(graph_ZS, point_types_ZS, spine_polys_ZS);
    double t3= GetCurrentTime();
    TThinningGraph graph_GH= DetectNodesFromThinningImg(img_thinning_GH, point_types_GH);
    ConnectNodes(graph_GH, point_types_GH);
    ThinningGraphToSpinePolys(graph_GH, point_types_GH, spine_polys_GH);
    double t4= GetCurrentTime();

    std::cout<<"Computation times:"<<endl
      <<"  ZHANGSUEN: "<<t1-t0<<" sec, "<<t3-t2<<" sec"<<endl
      <<"  GUOHALL: "  <<t2-t1<<" sec, "<<t4-t3<<" sec"<<endl;

    // Visualize results
    cv::Mat result_ZS(img.rows, img.cols, CV_8UC3), result_GH(img.rows, img.cols, CV_8UC3);
    cv::Mat in[] = {img_thinning_ZS, img_thinning_ZS, img_thinning_ZS};
    cv::Mat in2[] = {img_thinning_GH, img_thinning_GH, img_thinning_GH};
    int from_to[] = {0,0, 1,1, 2,2};
    cv::mixChannels(in, 3, &result_ZS, 1, from_to, 3);
    cv::mixChannels(in2, 3, &result_GH, 1, from_to, 3);
    result_ZS= 0.5*img + result_ZS;
    result_GH= 0.5*img + result_GH;
    DrawThinningGraph(result_ZS, graph_ZS, spine_polys_ZS);
    DrawThinningGraph(result_GH, graph_GH, spine_polys_GH);
    cv::imshow("Input", img_in);
    cv::imshow("Thinning ZHANGSUEN", result_ZS);
    cv::imshow("Thinning GUOHALL", result_GH);

    char c(cv::waitKey(500));
    if(c=='\x1b'||c=='q') break;
  }

  return 0;
}
#endif//LIBRARY
//-------------------------------------------------------------------------------------------
