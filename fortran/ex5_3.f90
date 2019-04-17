program hoge
  use omp_lib
implicit none
integer a(8),b(8),z
integer i,j
real e0,t0(2),e1,t1(2)
  double precision t20, t21
e0=etime(t0)
  t20= omp_get_wtime()
!$omp parallel do
!private(j,z)
do i=1,8
  do j=1,1000000
    z= j**1.2
    a(i)=z
    b(i)=z+1

!    a(i)=a(i)+1
!    b(i)=a(i)*a(i)

!    a(i)=j**1.2
!    b(i)=a(i)+1
  enddo
  !call sleep(1)
enddo
!$omp end parallel do
e1=etime(t1)
print *,"elapsed time=",e1-e0
  t21= omp_get_wtime()
  print *,"elapsed time(omp)=",t21-t20
print *, a
stop
end program hoge
