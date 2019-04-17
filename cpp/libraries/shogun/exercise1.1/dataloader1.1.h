#include <iostream>
#include <fstream>
#include <list>
#include <sstream>
#include <cassert>
#include <shogun/lib/DataType.h>

static void load_data_from_file(const char filename[], std::list<std::list<double> > &scaned, int max_sample)
{
  using namespace std;
  ifstream ifs(filename);
  if(!ifs)  {std::cerr<<"failed to load from: "<<filename<<std::endl; return;}
  std::cerr<<"loading: "<<filename<<std::endl;
  string line;
  int r(0);
  double value;
  while(getline(ifs,line,'\n'))
  {
    if(line=="") continue;
    scaned.push_back(std::list<double>());
    stringstream ss(line);
    while(ss>>value)
      scaned.back().push_back(value);
    ++r;
    if(max_sample>0 && r>=max_sample)  break;
  }
}

// load matrix from file
shogun::SGMatrix<float64_t> LoadMatFromFile(const char filename[], int &num, int &dim, int max_sample=-1)
{
  using namespace std;
  list<list<double> > scaned;
  load_data_from_file(filename, scaned, max_sample);
  num= scaned.size();
  dim= scaned.front().size();

  shogun::SGMatrix<float64_t> data(dim,num);
  int r= 0;
  for (list<list<double> >::const_iterator itr(scaned.begin()),last(scaned.end()); itr!=last; ++itr,++r)
  {
    int c(0);
    for (list<double>::const_iterator ditr(itr->begin()),dlast(itr->end()); ditr!=dlast; ++ditr,++c)
      data[r*dim+c]= *ditr;
  }
  return data;
}

// load vector from file (col-th column is loaded)
shogun::SGVector<float64_t> LoadVecFromFile(const char filename[], int &num, int &dim, int col=0, int max_sample=-1)
{
  using namespace std;
  list<list<double> > scaned;
  load_data_from_file(filename, scaned, max_sample);
  assert(col<(int)scaned.front().size());
  num= scaned.size();
  dim= 1;

  shogun::SGVector<float64_t> data(num);
  int r= 0;
  for (list<list<double> >::const_iterator itr(scaned.begin()),last(scaned.end()); itr!=last; ++itr,++r)
  {
    list<double>::const_iterator ditr(itr->begin());
    for (int c(0); c<col; ++c) ++ditr;
    data[r]= *ditr;
  }
  return data;
}
