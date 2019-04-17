program enshu2_1
  implicit none
  integer,parameter :: N=1000
  real :: r = 1.0
  real :: dx,x,y,z
  real :: ca_result= 0.0
  integer :: i,j
  dx = r/N
  do i=1, N
    x= dx*i
    do j=1, N
      y= dx*j
      if((x*x+y*y)<r*r) then
        z= sqrt(r*r-x*x-y*y)
        ca_result= ca_result + z*dx*dx
      endif
    enddo
  enddo
  ca_result= ca_result*8.0
  print *,"ca_result=", ca_result, "analytic result=",(4.0/3.0)*3.1415926535*r*r*r
  stop
  end program enshu2_1


