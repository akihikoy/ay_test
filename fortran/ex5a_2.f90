program hoge
use mpi
implicit none
integer ierr,rank,rdata
integer status(MPI_STATUS_SIZE)

call MPI_INIT(ierr)
call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)

if(rank==0) then
  call MPI_RECV(rdata,1,MPI_INTEGER,1,0,MPI_COMM_WORLD,status,ierr)
  print *,'received:',rdata
else if(rank==1) then
  call MPI_SEND(rank,1,MPI_INTEGER,0,0,MPI_COMM_WORLD,ierr)
endif

call MPI_FINALIZE(ierr)

stop
end program hoge
