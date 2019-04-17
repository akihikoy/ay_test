#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <iostream>

typedef boost::numeric::ublas::vector<double> v_type;

using namespace std;

// "Eager" evaluation cross product
template <class V1, class V2>
boost::numeric::ublas::vector<typename boost::numeric::ublas::promote_traits<typename V1::value_type,
                                                                             typename V2::value_type>::promote_type>
cross_prod(const V1& lhs, const V2& rhs)
{
  BOOST_UBLAS_CHECK(lhs.size() == 3, boost::numeric::ublas::external_logic());
  BOOST_UBLAS_CHECK(rhs.size() == 3, boost::numeric::ublas::external_logic());

  typedef typename boost::numeric::ublas::promote_traits<typename V1::value_type,
                                                         typename V2::value_type>::promote_type promote_type;

  boost::numeric::ublas::vector<promote_type> temporary(3);

  temporary(0) = lhs(1) * rhs(2) - lhs(2) * rhs(1);
  temporary(1) = lhs(2) * rhs(0) - lhs(0) * rhs(2);
  temporary(2) = lhs(0) * rhs(1) - lhs(1) * rhs(0);

  return temporary;
}

int main()
{
  v_type a(3),b(3);
  cout<<outer_prod(a,b)<<endl;
  cout<<cross_prod(a,b)<<endl;
  return 0;
}
