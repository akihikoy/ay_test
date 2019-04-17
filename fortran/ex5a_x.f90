program hoge
use mpi
implicit none
integer ierr,rank,num_rank,i,sum,total
integer,parameter:: imax=100000
real rnd(2)

!Initialize random seed
call random_seed_clock()

call MPI_INIT(ierr)
call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)

sum=0
do i=1,imax
  call random_number(rnd)
  if(rnd(1)**2+rnd(2)**2<1.0) then
    sum=sum+1
  endif
end do
print *,rank,sum

call MPI_REDUCE(sum,total,1,MPI_INTEGER,MPI_SUM,0,MPI_COMM_WORLD,ierr)
call MPI_REDUCE(1,num_rank,1,MPI_INTEGER,MPI_SUM,0,MPI_COMM_WORLD,ierr)
if(rank==0) then
  print *,'num_rank=',num_rank
  print *,'pi=',4.0*dble(total)/(imax*num_rank)
end if

call MPI_FINALIZE(ierr)

stop
end program hoge


subroutine random_seed_clock()
  implicit none
  integer :: nseed, clock
  integer, allocatable :: seed(:)

  call system_clock(clock)

  call random_seed(size=nseed)
  allocate(seed(nseed))

  seed = clock
  call random_seed(put=seed)

  deallocate(seed)
end subroutine random_seed_clock
