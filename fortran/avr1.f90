program avr1
  implicit none
  real :: avr,max,min
  !real dat(3)!=(/180.,154.,172./)
  integer :: n=3
  real, dimension(3) :: dat=(/180.,154.,172./)
  print *, dat
  !call stat(dat, n, avr, max, min)
  call statt(dat, n, avr, max, min)
  print *, avr, max, min
  !call stat1(dat, n)
  stop
end program avr1

subroutine statt(dat, n, avr, max, min)
  implicit none
  real, dimension(n) :: dat
  integer :: n
  real :: avr,max,min
  avr= sum(dat)/n
  max= maxval(dat)
  min= minval(dat)
end subroutine statt

subroutine stat1(dat, n)
  implicit none
  real, dimension(n) :: dat
  integer :: n
  print *, dat
end subroutine stat1

