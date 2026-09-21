! Microbench reproducing the ABI of WRF's micro tile loop (!$OMP PARALLEL DO, no SCHEDULE clause, 13 tiles).
! Iterations 1-7 cost 3 units, 8-13 cost 1 unit, so a static 7/6 split gives thread 0 21 units and thread 1 6 units;
! any mechanism that moves iteration ownership shows up as a different owner string and a busy ratio near 1.
! Build with -DRUNTIME_SCHED for the positive control (SCHEDULE(RUNTIME) obeys OMP_SCHEDULE).
program sched_bench
  use omp_lib
  implicit none
  integer, parameter :: nt = 13, nrep = 5
  integer :: ij, r, owner(nt)
  double precision :: tw(nt), res(nt), busy(0:1), wall(nrep), t0, tr
  character(len=32) :: mode, build
  character(len=nt) :: own
  call get_command_argument(1, mode)
#ifdef RUNTIME_SCHED
  build = 'runtime_clause'
#else
  build = 'no_clause'
#endif
  busy = 0d0
  do r = 1, nrep
    tr = omp_get_wtime()
#ifdef RUNTIME_SCHED
    !$OMP PARALLEL DO SCHEDULE(RUNTIME) PRIVATE(ij, t0)
#else
    !$OMP PARALLEL DO PRIVATE(ij, t0)
#endif
    do ij = 1, nt
      t0 = omp_get_wtime()
      res(ij) = work(ij)
      owner(ij) = omp_get_thread_num()
      tw(ij) = omp_get_wtime() - t0
    end do
    !$OMP END PARALLEL DO
    wall(r) = omp_get_wtime() - tr
    do ij = 1, nt
      busy(min(owner(ij), 1)) = busy(min(owner(ij), 1)) + tw(ij)
    end do
  end do
  do ij = 1, nt
    write(own(ij:ij), '(i1)') owner(ij)
  end do
  write(*, '(a,a,a,a,a,a,a,f8.1,a,f8.1,a,f6.2,a,f8.1,a,es12.5)') 'RESULT build=', trim(build), ' mode=', trim(mode), &
    ' owners=', own, ' busy0_ms=', 1d3*busy(0)/nrep, ' busy1_ms=', 1d3*busy(1)/nrep, &
    ' ratio=', max(busy(0), busy(1))/max(min(busy(0), busy(1)), 1d-9), ' region_ms=', 1d3*sum(wall)/nrep, ' check=', sum(res)
contains
  double precision function work(ij)
    integer, intent(in) :: ij
    integer :: i, n
    n = merge(3, 1, ij <= 7) * 6000000
    work = 0d0
    do i = 1, n
      work = work + sqrt(dble(i) + dble(ij))
    end do
  end function work
end program sched_bench
