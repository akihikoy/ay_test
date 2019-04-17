program stest1
  implicit none
  real x,y
  x=5
  y=100
  print *,x,y
  call subr(x,y,10)
  print *,x,y
end program stest1

subroutine subr(x,y,n)
  implicit none
  real x,y
  integer n
  x = n
  y = y*x
end subroutine subr


