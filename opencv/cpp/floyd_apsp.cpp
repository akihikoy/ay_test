//-------------------------------------------------------------------------------------------
/*! \file    floyd_apsp.cpp
    \brief   Floyd's all-pairs-shortest-path (APSP) algorithm.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Jul.19, 2021

Reference: https://www.cs.rochester.edu/u/nelson/courses/csc_173/graphs/apsp.html

g++ -g -Wall -O2 -o floyd_apsp.out floyd_apsp.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui -I/usr/include/opencv4
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <list>
#include <iostream>
//-------------------------------------------------------------------------------------------
namespace loco_rabbits
{
}
//-------------------------------------------------------------------------------------------
using namespace std;
// using namespace boost;
using namespace loco_rabbits;
//-------------------------------------------------------------------------------------------
// #define print(var) PrintContainer((var), #var"= ")
// #define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

struct TFloydAPSPEdge
{
  int Node1;
  int Node2;
  double Cost;
  TFloydAPSPEdge(int n1, int n2, const  double &c) : Node1(n1), Node2(n2), Cost(c) {}
};

//Floyd's all-pairs-shortest-path (APSP) algorithm.
//  N: Number of nodes.
//  edges: List of edge and cost values: (node1,node2,cost).
void FloydAPSP(int N, const std::list<TFloydAPSPEdge> &edges, cv::Mat &D, cv::Mat &P, const double &inf_cost=1.0e9)
{
  //C: cost matrix.
  cv::Mat C= cv::Mat::ones(N, N, CV_64F)*inf_cost;
  double c;
  for(std::list<TFloydAPSPEdge>::const_iterator eitr(edges.begin()),eitr_end(edges.end()); eitr!=eitr_end; ++eitr)
  {
    c= std::min(std::min(C.at<double>(eitr->Node1,eitr->Node2),C.at<double>(eitr->Node2,eitr->Node1)),eitr->Cost);
    C.at<double>(eitr->Node1,eitr->Node2)= c;
    C.at<double>(eitr->Node2,eitr->Node1)= c;
  }
  //D: matrix containing lowest cost.
  //P: matrix containing a via point on the shortest path.
  D= cv::Mat::zeros(N, N, CV_64F);
  P= cv::Mat::zeros(N, N, CV_32S);
  for(int i(0); i<N; ++i)
  {
    for(int j(0); j<N; ++j)
    {
      D.at<double>(i,j)= C.at<double>(i,j);
      P.at<int>(i,j)= C.at<double>(i,j)<inf_cost ? -1 : -2;
    }
    D.at<double>(i,i)= 0.0;
    P.at<int>(i,i)= -1;
  }
  for(int k(0); k<N; ++k)
  {
    for(int i(0); i<N; ++i)
    {
      for(int j(0); j<N; ++j)
      {
        if(D.at<double>(i,k) + D.at<double>(k,j) < D.at<double>(i,j))
        {
          D.at<double>(i,j)= D.at<double>(i,k) + D.at<double>(k,j);
          P.at<int>(i,j)= k;
        }
      }
    }
  }
}
//-------------------------------------------------------------------------------------------

// From the output path matrix P of FloydAPSP, find a shortest path between two nodes.
// _flag: Internally used in recursive call. Do not use.
bool ShortestPath(int n1, int n2, const cv::Mat &P, std::list<int> &path, bool _flag=false)
{
  int k= P.at<int>(n1,n2);
  if(k==-1)
  {
    if(!_flag)  path.push_back(n1);
    path.push_back(n2);
    return true;
  }
  if(k==-2)
  {
    path.clear();
    return false;
  }
  bool p1= ShortestPath(n1,k,P, path, _flag);
  bool p2= ShortestPath(k,n2,P, path, true);
  if(p1 && p2)  return true;
  path.clear();
  return false;
}
//-------------------------------------------------------------------------------------------

// From the output path matrix P of FloydAPSP, find isolated sub graphs.
std::vector<std::vector<int> > IsolatedGraphs(const cv::Mat &P)
{
  std::vector<std::vector<int> > iso_graphs;
  std::list<int> remains;
  for(int n1(0); n1<P.rows; ++n1)  remains.push_back(n1);
  while(remains.size()>0)
  {
    int n1= remains.front();
    std::vector<int> sub_graph;
    for(std::list<int>::iterator n2itr(remains.begin()); n2itr!=remains.end();)
    {
// std::cerr<<" "<<*n2itr<<" "<<P.at<int>(n1,*n2itr)<<std::endl;
      if(P.at<int>(n1,*n2itr)!=-2)
      {
        sub_graph.push_back(*n2itr);
        n2itr= remains.erase(n2itr);
      }
      else  ++n2itr;
    }
    iso_graphs.push_back(sub_graph);
  }
  return iso_graphs;
}
//-------------------------------------------------------------------------------------------

#ifndef LIBRARY
int main(int argc, char**argv)
{
  /*
  Definition of a graph:
              |--2--|       |-----3------|
              |     |       |            |
      n1--1--n3--1--n4--2--n5--1--n6--1--n7--1--n9
             |                    |
      n2--2--|             n8--2--|
  */
  int N= 9;  // Number of nodes.
  std::list<TFloydAPSPEdge>  edges;  // List of edge and cost values: (node1,node2,cost).
  edges.push_back(TFloydAPSPEdge(0,2,1.0));
  edges.push_back(TFloydAPSPEdge(1,2,2.0));
  edges.push_back(TFloydAPSPEdge(2,3,1.0));
  edges.push_back(TFloydAPSPEdge(2,3,2.0));
  edges.push_back(TFloydAPSPEdge(3,4,2.0));   // NOTE: Comment out this edge to make the graph two isolated parts.
  edges.push_back(TFloydAPSPEdge(4,5,1.0));
  edges.push_back(TFloydAPSPEdge(5,6,1.0));
  edges.push_back(TFloydAPSPEdge(4,6,3.0));
  edges.push_back(TFloydAPSPEdge(5,7,2.0));
  edges.push_back(TFloydAPSPEdge(6,8,1.0));

  cv::Mat D,P;
  FloydAPSP(N, edges, D, P);
  std::cout<<"Lowest cost matrix:"<<std::endl<<D<<std::endl;
  std::cout<<"Via points on the shortest path matrix:"<<std::endl<<P<<std::endl;
  std::vector<std::vector<int> > iso_graphs= IsolatedGraphs(P);
  std::cout<<"Isolated graphs: ";
  for(int i(0),i_end(iso_graphs.size());i<i_end;++i)
  {
    std::cout<<"[";
    for(int j(0),j_end(iso_graphs[i].size());j<j_end;++j)
      std::cout<<" "<<iso_graphs[i][j];
    std::cout<<"]";
  }
  std::cout<<std::endl;

  while(true)
  {
    std::cout<<"Type two node indexes (starting from 1) separating with space (0 0 to quit):"<<std::endl;
    std::cout<<" > ";
    int n1(0),n2(0);
    std::cin>>n1>>n2; --n1; --n2;
    if(n1<0 || n2<0 || n1>=N || n2>=N)  break;
    std::list<int> path;
    bool res= ShortestPath(n1, n2, P, path);
    std::cout<<"  Shortest path:";
    for(std::list<int>::const_iterator itr(path.begin()),itr_end(path.end()); itr!=itr_end; ++itr)
      std::cout<<" "<<*itr+1;
    if(!res)  std::cout<<"not found";
    std::cout<<" / Cost: "<<D.at<double>(n1,n2)<<std::endl;
  }
  return 0;
}
#endif//LIBRARY
//-------------------------------------------------------------------------------------------
