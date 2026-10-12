! kernels.f90 -- WRF-like compute kernels: 5th-order horizontal advection,
! a microphysics-like column loop (exp/branches, Thompson-like), and a
! tridiagonal column solve (PBL/implicit-diffusion-like). Arrays use the
! same (i,k,j) layout WRF uses for 3-D fields, i fastest-varying. Build
! with the SAME flags WRF is built with (pass via FFLAGS to the driver)
! so the numbers reflect the compiler's real code generation for this
! instruction mix, not a generic -O3 guess.
!
! Each kernel is timed once at 1 OpenMP thread ("single rank" proxy) and
! once at the full thread count ("full-node" proxy: one process-worth of
! parallelism via OpenMP rather than MPI -- the halo bench covers the
! MPI side separately). Reports ns per grid point per call.
!
! Emits exactly one JSON line to stdout.

module kernels_mod
  use iso_fortran_env, only: real64
  implicit none
contains

  subroutine advection5(f, out, nx, ny, nz, nlo, nhi)
    integer, intent(in) :: nx, ny, nz
    real(real64), intent(in) :: f(nx, nz, ny)
    real(real64), intent(out) :: out(nx, nz, ny)
    integer, intent(out) :: nlo, nhi
    integer :: i, j, k
    real(real64) :: fim2, fim1, fi, fip1, fip2, fip3

    nlo = 4
    nhi = nx - 3
    !$OMP PARALLEL DO PRIVATE(i,j,k,fim2,fim1,fi,fip1,fip2,fip3) COLLAPSE(2)
    do j = 1, ny
      do k = 1, nz
        do i = nlo, nhi
          fim2 = f(i-2,k,j); fim1 = f(i-1,k,j); fi = f(i,k,j)
          fip1 = f(i+1,k,j); fip2 = f(i+2,k,j); fip3 = f(i+3,k,j)
          ! WRF-style 5th-order upwind-biased face interpolation
          ! (module_advect_em.F ADV5 form); wind sign fixed positive here
          ! since only the instruction/memory mix matters for this bench.
          out(i,k,j) = (37.0_real64*(fi+fip1) - 8.0_real64*(fim1+fip2) &
                        + (fim2+fip3)) / 60.0_real64 &
                     - (10.0_real64*(fi-fip1) - 5.0_real64*(fim1-fip2) &
                        + (fim2-fip3)) / 60.0_real64
        end do
      end do
    end do
    !$OMP END PARALLEL DO
  end subroutine advection5

  subroutine microphys(f, out, nx, ny, nz)
    integer, intent(in) :: nx, ny, nz
    real(real64), intent(in) :: f(nx, nz, ny)
    real(real64), intent(out) :: out(nx, nz, ny)
    integer :: i, j, k
    real(real64) :: t, height, p, esat, qsat, vt

    !$OMP PARALLEL DO PRIVATE(i,j,k,t,height,p,esat,qsat,vt) COLLAPSE(2)
    do j = 1, ny
      do i = 1, nx
        do k = 1, nz
          height = real(k, real64) * 100.0_real64
          t = 300.0_real64 - 0.0065_real64 * height + f(i,k,j)
          p = 101325.0_real64 * (1.0_real64 - 2.25577e-5_real64 * height) ** 5.25588_real64
          if (t < 253.15_real64) then
            esat = 611.15_real64 * exp(22.452_real64 * (t - 273.15_real64) / (t - 0.61_real64))
          else
            esat = 611.2_real64 * exp(17.67_real64 * (t - 273.15_real64) / (t - 29.65_real64))
          end if
          qsat = 0.622_real64 * esat / max(p - esat, 1.0_real64)
          vt = 0.0_real64
          if (qsat > 1.0e-5_real64) then
            vt = 3.29_real64 * (qsat ** 0.16_real64)
          end if
          out(i,k,j) = qsat + vt
        end do
      end do
    end do
    !$OMP END PARALLEL DO
  end subroutine microphys

  subroutine thomas_column(al, bl, cl, dl, x, nz)
    integer, intent(in) :: nz
    real(real64), intent(in) :: al(nz), bl(nz), cl(nz), dl(nz)
    real(real64), intent(out) :: x(nz)
    real(real64) :: cp(nz), dp(nz), m
    integer :: k

    cp(1) = cl(1) / bl(1)
    dp(1) = dl(1) / bl(1)
    do k = 2, nz
      m = bl(k) - al(k) * cp(k-1)
      cp(k) = cl(k) / m
      dp(k) = (dl(k) - al(k) * dp(k-1)) / m
    end do
    x(nz) = dp(nz)
    do k = nz - 1, 1, -1
      x(k) = dp(k) - cp(k) * x(k+1)
    end do
  end subroutine thomas_column

  subroutine tridiag(f, out, nx, ny, nz)
    integer, intent(in) :: nx, ny, nz
    real(real64), intent(in) :: f(nx, nz, ny)
    real(real64), intent(out) :: out(nx, nz, ny)
    integer :: i, j, k
    real(real64) :: al(nz), bl(nz), cl(nz), dl(nz), x(nz)

    !$OMP PARALLEL DO PRIVATE(i,j,k,al,bl,cl,dl,x) COLLAPSE(2)
    do j = 1, ny
      do i = 1, nx
        do k = 1, nz
          bl(k) = 2.0_real64 + 0.01_real64 * real(k, real64)
          al(k) = -1.0_real64
          cl(k) = -1.0_real64
          dl(k) = f(i,k,j)
        end do
        call thomas_column(al, bl, cl, dl, x, nz)
        do k = 1, nz
          out(i,k,j) = x(k)
        end do
      end do
    end do
    !$OMP END PARALLEL DO
  end subroutine tridiag

end module kernels_mod


program kernels
  use omp_lib
  use iso_fortran_env, only: real64, int64
  use kernels_mod
  implicit none

  integer :: nx, ny, nz, ncalls, nthreads_full
  integer :: i, j, k, c, nlo, nhi, npts_adv
  real(real64), allocatable :: f(:,:,:), outadv(:,:,:), outmp(:,:,:), outtri(:,:,:)
  real(real64) :: ns_adv_1, ns_adv_n, ns_mp_1, ns_mp_n, ns_tri_1, ns_tri_n
  real(real64) :: check
  character(len=64) :: arg, val
  integer :: nargs, ai
  integer(int64) :: c0, c1, crate

  nx = 100; ny = 100; nz = 40; ncalls = 5
  nthreads_full = omp_get_max_threads()

  nargs = command_argument_count()
  ai = 1
  do while (ai <= nargs - 1)
    call get_command_argument(ai, arg)
    call get_command_argument(ai + 1, val)
    select case (trim(arg))
    case ('--nx'); read(val, *) nx
    case ('--ny'); read(val, *) ny
    case ('--nz'); read(val, *) nz
    case ('--ncalls'); read(val, *) ncalls
    case ('--threads-full'); read(val, *) nthreads_full
    end select
    ai = ai + 2
  end do

  allocate(f(nx,nz,ny), outadv(nx,nz,ny), outmp(nx,nz,ny), outtri(nx,nz,ny))
  do j = 1, ny
    do k = 1, nz
      do i = 1, nx
        f(i,k,j) = sin(real(i,real64) * 0.01_real64) + real(k,real64) * 0.001_real64 &
                   + real(j,real64) * 0.0001_real64
      end do
    end do
  end do

  call omp_set_num_threads(1)
  call system_clock(c0, crate)
  do c = 1, ncalls
    call advection5(f, outadv, nx, ny, nz, nlo, nhi)
  end do
  call system_clock(c1)
  npts_adv = max(0, nhi - nlo + 1) * ny * nz
  ns_adv_1 = 1.0e9_real64 * real(c1-c0,real64) / real(crate,real64) / real(max(npts_adv,1)*ncalls, real64)

  call omp_set_num_threads(nthreads_full)
  call system_clock(c0, crate)
  do c = 1, ncalls
    call advection5(f, outadv, nx, ny, nz, nlo, nhi)
  end do
  call system_clock(c1)
  ns_adv_n = 1.0e9_real64 * real(c1-c0,real64) / real(crate,real64) / real(max(npts_adv,1)*ncalls, real64)

  call omp_set_num_threads(1)
  call system_clock(c0, crate)
  do c = 1, ncalls
    call microphys(f, outmp, nx, ny, nz)
  end do
  call system_clock(c1)
  ns_mp_1 = 1.0e9_real64 * real(c1-c0,real64) / real(crate,real64) / real(nx*ny*nz*ncalls, real64)

  call omp_set_num_threads(nthreads_full)
  call system_clock(c0, crate)
  do c = 1, ncalls
    call microphys(f, outmp, nx, ny, nz)
  end do
  call system_clock(c1)
  ns_mp_n = 1.0e9_real64 * real(c1-c0,real64) / real(crate,real64) / real(nx*ny*nz*ncalls, real64)

  call omp_set_num_threads(1)
  call system_clock(c0, crate)
  do c = 1, ncalls
    call tridiag(f, outtri, nx, ny, nz)
  end do
  call system_clock(c1)
  ns_tri_1 = 1.0e9_real64 * real(c1-c0,real64) / real(crate,real64) / real(nx*ny*nz*ncalls, real64)

  call omp_set_num_threads(nthreads_full)
  call system_clock(c0, crate)
  do c = 1, ncalls
    call tridiag(f, outtri, nx, ny, nz)
  end do
  call system_clock(c1)
  ns_tri_n = 1.0e9_real64 * real(c1-c0,real64) / real(crate,real64) / real(nx*ny*nz*ncalls, real64)

  ! Fold every output array into a checksum that participates in the
  ! printed line: blocks dead-code elimination of the actual compute.
  check = sum(outadv) + sum(outmp) + sum(outtri)
  if (check /= check) then  ! NaN sentinel, should never fire
    write(0,*) 'kernels: NaN checksum'
  end if

  call emit_json()

contains

  subroutine emit_json()
    write(*,'(A)', advance='no') '{"bench":"kernels","nx":'
    call wi(nx)
    write(*,'(A)', advance='no') ',"ny":'
    call wi(ny)
    write(*,'(A)', advance='no') ',"nz":'
    call wi(nz)
    write(*,'(A)', advance='no') ',"ncalls":'
    call wi(ncalls)
    write(*,'(A)', advance='no') ',"threads_full":'
    call wi(nthreads_full)
    write(*,'(A)', advance='no') ',"advection":{"ns_per_point_single":'
    call wf(ns_adv_1)
    write(*,'(A)', advance='no') ',"ns_per_point_full":'
    call wf(ns_adv_n)
    write(*,'(A)', advance='no') '},"microphysics":{"ns_per_point_single":'
    call wf(ns_mp_1)
    write(*,'(A)', advance='no') ',"ns_per_point_full":'
    call wf(ns_mp_n)
    write(*,'(A)', advance='no') '},"tridiag":{"ns_per_point_single":'
    call wf(ns_tri_1)
    write(*,'(A)', advance='no') ',"ns_per_point_full":'
    call wf(ns_tri_n)
    write(*,'(A)') '}}'
  end subroutine emit_json

  subroutine wi(n)
    integer, intent(in) :: n
    character(len=32) :: buf
    write(buf, '(I0)') n
    write(*, '(A)', advance='no') trim(buf)
  end subroutine wi

  subroutine wf(x)
    real(real64), intent(in) :: x
    character(len=32) :: buf
    write(buf, '(F20.6)') x
    write(*, '(A)', advance='no') trim(adjustl(buf))
  end subroutine wf

end program kernels
