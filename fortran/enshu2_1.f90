program enshu2_1
  implicit none
  integer,parameter :: N=100
  real :: a = 1.0
  real :: dx
  real :: f = 0.0
  real :: ca_result= 0.0
  integer :: i
  dx = a/N
  do i=1, N
    f= a**2 - (dx*i)**2
    ca_result= ca_result + f*dx
  enddo
  print *,"ca_result=", ca_result, "analytic result=", 2.0/3.0
  stop
  end program enshu2_1


