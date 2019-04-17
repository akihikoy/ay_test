program enshu_1
  implicit none
  integer, parameter :: N=100
  integer :: i
  real :: ca_result
  real area(N),f(N)
  real ::a
  real :: dx
  a = 1.0
  dx = a/N
  ca_result = 0.0
  do i=1,N
    f(i)=a**2 - (dx*i)**2
    area(i)=f(i)*dx
    ca_result=ca_result+area(i)
  enddo
  print *,"ca_result= ",ca_result,"theoretical value=",2./3.
  stop
  end program enshu_1

