#include <fstream>
#include <list>
#include <sstream>
#include <cassert>

// load data from file; memory for data is allocated
// if col==-1, whole columns are loaded, otherwise a specific column is loaded
void LoadDataFromFile(const char filename[], double* (&data), int &num, int &dim, int col=-1, int max_sample=-1)
{
  using namespace std;
  list<list<double> > scaned;
  ifstream ifs(filename);
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

  num= scaned.size();
  dim= (col<0) ? scaned.front().size() : 1;
  if(col>=0) assert(col<(int)scaned.front().size());
  data= new double[num*dim];
  r= 0;
  for (list<list<double> >::const_iterator itr(scaned.begin()),last(scaned.end()); itr!=last; ++itr,++r)
  {
    if(col<0)
    {
      int c(0);
      for (list<double>::const_iterator ditr(itr->begin()),dlast(itr->end()); ditr!=dlast; ++ditr,++c)
        data[r*dim+c]= *ditr;
    }
    else
    {
      list<double>::const_iterator ditr(itr->begin());
      for (int c(0); c<col; ++c) ++ditr;
      data[r]= *ditr;
    }
  }
}
