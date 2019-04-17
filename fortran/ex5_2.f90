program hoge
implicit none
integer a(8),b(8)
integer i,j
real e0,t0(2),e1,t1(2)
e0=etime(t0)
do i=1,8
  a(i)=i
  if(i>1)  a(i)=a(i-1)
  do j=1,100000000
    a(i)=a(i)+1
    b(i)=a(i)*a(i)
  enddo
  !call sleep(1)
enddo
e1=etime(t1)
print *,"elapsed time=",e1-e0
print *, a
stop
end program hoge
