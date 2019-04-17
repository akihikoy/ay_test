program hoge
use omp_lib
implicit none
integer a(8),b(8),z
integer i,j
double precision t20, t21
t20= omp_get_wtime()
!$omp parallel do
do i=1,8
  do j=1,1000000
    z= j**1.2
    a(i)=z
    b(i)=z+1
  enddo
enddo
!$omp end parallel do
t21= omp_get_wtime()
print *,"elapsed time(omp)=",t21-t20
stop
end program hoge
