program hoge
use mpi
implicit none
integer ierr,rank

call MPI_INIT(ierr)
call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)
print *,'my rank is', rank
if(rank==0)  print *,'I am zero'
call MPI_FINALIZE(ierr)

stop
end program hoge
