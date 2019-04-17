module gtest
  implicit none
  real :: param
end module gtest

program global1
  use gtest
  implicit none
  param= 100
  call subr
  print *, param
end program global1

subroutine subr
  use gtest
  implicit none
  param= 200
end subroutine subr


