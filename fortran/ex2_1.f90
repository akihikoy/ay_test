program ex2_1
  implicit none
  real :: a(3), b(3,4), c(3,4)
  integer :: i
  a=1
  b=2
  c=3
  !b(0,0)=5 !index out of range
  b(1,1)=5
  b(2,2)=5
  b(2:3,3:4)=9
  c(1,:)=0
  print *, "a=", a
  print *, "b=", b
  print *, "b(1,:)=", b(1,:)
  print *, "b(2,:)=", b(2,:)
  print *, "b(3,:)=", b(3,:)
  print *, "c=", c
  do i=1,3,1
    print *,"c(",i,",:)=", c(i,:)
  enddo
  end program ex2_1



