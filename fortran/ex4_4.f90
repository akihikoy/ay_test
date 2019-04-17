program hoge
implicit none
real x,y
integer n
open(11,file="input.dat")
read(11,*) x,y
read(11,*) n
close(11)
print *, x,y,n
stop
end program hoge
