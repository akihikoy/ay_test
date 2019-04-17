program hoge
use mpi
implicit none
integer ierr,rank,i,sum,total

call MPI_INIT(ierr)
call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)

sum=0
do i=rank*10,rank*10+9
  sum=sum+i
end do

call MPI_REDUCE(sum,total,1,MPI_INTEGER,MPI_SUM,0,MPI_COMM_WORLD,ierr)
if(rank==0) then
  print *,'total=',total
end if

call MPI_FINALIZE(ierr)

stop
end program hoge
