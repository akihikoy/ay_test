program ex42
implicit none
real a(3,3)
integer i,j,k
do i=1,3
do j=1,3
a(i,j) = i
enddo
print *, (a(i,k), k=1,3)
enddo
stop
end program ex42

