program enshu2_1
  implicit none
  integer,parameter :: N=1000
  real :: r = 1.0
  real :: dx,x,f
  real :: ca_result= 0.0
  integer :: i
  dx = r/N
  do i=1, N
    x= dx*i
    f= sqrt(r*r-x*x)
    ca_result= ca_result + f*dx
  enddo
  ca_result= ca_result*4.0
  print *,"ca_result=", ca_result, "analytic result=",3.1415926535*r*r
  stop
  end program enshu2_1


