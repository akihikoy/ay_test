//-------------------------------------------------------------------------------------------
/*! \file    lib_py_test1.cpp
    \brief   Wrapper of lib_cpp_test1 for Python binding.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Aug.15, 2022

Compile:

g++ -std=c++11 -O3 -Wall -shared -fPIC -I/usr/include/python2.7 `python -m pybind11 --includes` lib_py_test1.cpp lib_cpp_test1.cpp -o lib_py_test1`python3-config --extension-suffix`
ln -s lib_py_test1.cpython-36m-x86_64-linux-gnu.so lib_py_test1.so

Run:
python
> import lib_py_test1
> lib_py_test1.Add(100,200)
300
> lib_py_test1.Add(100.,200.)
TypeError
> lib_py_test1.Add(100,y=100)
200
> lib_py_test1.Add(100)
101
> lib_py_test1.VecConcatenate([1,2,3,4],[5,6,7,8])
[1, 2, 3, 4, 5, 6, 7, 8]
> lib_py_test1.VecConcatenate(np.array([1,2,3,4]),np.array([5,6,7,8]))
[1, 2, 3, 4, 5, 6, 7, 8]
> lib_py_test1.MatAdd([[1,2],[3,4]],[[5,6],[7,8]])
[[6, 8], [10, 12]]
> lib_py_test1.MatAdd(np.array([[1,2],[3,4]]),np.array([[5,6],[7,8]]))
[[6, 8], [10, 12]]
> lib_py_test1.MatAdd(np.array([1,2,3,4]).reshape(-1,2),np.array([5,6,7,8]).reshape(-1,2))
[[6, 8], [10, 12]]
> type(lib_py_test1.MatAdd(np.array([1,2,3,4]).reshape(-1,2),np.array([5,6,7,8]).reshape(-1,2)))
< type 'list' >

> import lib_py_test1
> test= lib_py_test1.TTest(10,20,)
> test.X()
10
> test.XY()
[10, 20]
> test.Sum()
30
> test.Y()= 30
SyntaxError: can't assign to function call
> tt= lib_py_test1.TTestTest()
> tt.test1.XY()
[10, 20]
> tt.test2.XY()
[30, 40]
*/
//-------------------------------------------------------------------------------------------
// #include "lib_py_test1.h"
#include "lib_cpp_test1.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
//-------------------------------------------------------------------------------------------
namespace test_py
{

using namespace test;

namespace py= pybind11;
PYBIND11_PLUGIN(lib_py_test1)
{
  py::module m("lib_py_test1", "Wrapper of lib_cpp_test1 with pybind11");
  m.def("Add", &Add, "Add two integers.",
        py::arg("x"), py::arg("y")=1);
  m.def("VecConcatenate", &VecConcatenate);
  m.def("MatAdd", &MatAdd);

  py::class_<TTest>(m, "TTest")
    .def(py::init<int,int>())
    .def("X", static_cast<int& (TTest::*)()>(&TTest::X))
    .def("X", static_cast<const int& (TTest::*)() const>(&TTest::X))
    .def("Y", static_cast<int& (TTest::*)()>(&TTest::Y))
    .def("Y", static_cast<const int& (TTest::*)() const>(&TTest::Y))
    .def("Sum", &TTest::Sum)
    .def("XY", &TTest::XY);

  py::class_<TTestTest>(m, "TTestTest")
    .def(py::init<>())
    .def_readwrite("test1", &TTestTest::test1)
    .def_readwrite("test2", &TTestTest::test2);
}
//-------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------
}  // end of test_py
//-------------------------------------------------------------------------------------------

