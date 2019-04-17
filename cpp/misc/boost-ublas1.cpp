#include <algorithm>
#include <vector>
#include <boost/numeric/ublas/banded.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#define BOOST_NUMERIC_BINDINGS_USE_CLAPACK
#include <boost/numeric/bindings/lapack/gesvd.hpp>
#include <boost/numeric/bindings/traits/std_vector.hpp>
#include <boost/numeric/bindings/traits/ublas_matrix.hpp>
#undef  BOOST_NUMERIC_BINDINGS_USE_CLAPACK

namespace math {
    void svd(const boost::numeric::ublas::matrix<double>& A, boost::numeric::ublas::matrix<double>& U, boost::numeric::ublas::diagonal_matrix<double>& D, boost::numeric::ublas::matrix<double>& VT);
}

void math::svd(const boost::numeric::ublas::matrix<double>& A, boost::numeric::ublas::matrix<double>& U, boost::numeric::ublas::diagonal_matrix<double>& D, boost::numeric::ublas::matrix<double>& VT)
{
    namespace ublas = boost::numeric::ublas;

    std::vector<double>                        s((std::min)(A.size1(), A.size2()));
    ublas::matrix<double, ublas::column_major> CA(A), CU(A.size1(), A.size1()), CVT(A.size2(), A.size2());
    int                                        info;

    info = boost::numeric::bindings::lapack::gesvd('A', 'A', CA, s, CU, CVT);
    BOOST_UBLAS_CHECK(info == 0, ublas::internal_logic());

    ublas::matrix<double>          CCU(CU), CCVT(CVT);
    ublas::diagonal_matrix<double> CD(A.size1(), A.size2());

    for (std::size_t i = 0; i < s.size(); ++i) {
        CD(i, i) = s[i];
    }

#if BOOST_UBLAS_TYPE_CHECK
    BOOST_UBLAS_CHECK(
        ublas::detail::expression_type_check(ublas::prod(ublas::matrix<double>(ublas::prod(CCU, CD)), CCVT), A),
        ublas::internal_logic()
    );
#endif

    U.assign_temporary(CCU);
    D.assign_temporary(CD);
    VT.assign_temporary(CCVT);
}


#include <boost/numeric/ublas/banded.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>


int main()
{
    namespace ublas = boost::numeric::ublas;

    ublas::matrix<double> m(4, 5);

    m(0, 0) = 1; m(0, 1) = 0; m(0, 2) = 0; m(0, 3) = 0; m(0, 4) = 2;
    m(1, 0) = 0; m(1, 1) = 0; m(1, 2) = 3; m(1, 3) = 0; m(1, 4) = 0;
    m(2, 0) = 0; m(2, 1) = 0; m(2, 2) = 0; m(2, 3) = 0; m(2, 4) = 0;
    m(3, 0) = 0; m(3, 1) = 4; m(3, 2) = 0; m(3, 3) = 0; m(3, 4) = 0;

    ublas::matrix<double> U, VT;
    ublas::diagonal_matrix<double> D;

    math::svd(m, U, D, VT); // m = U x D x VT

    // 以下 U, D, VT の表示
    // 計算結果を http://en.wikipedia.org/wiki/Singular_value_decomposition#Example と比較してみましょう
    std::cout << U << "\n";
    std::cout << D << "\n";
    std::cout << VT << std::endl;

    return 0;
}

