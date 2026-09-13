! Minimal unit fixture only; deliberately NOT a native WRF module/ABI test.
module module_domain_type
  implicit none
  type fieldlist
    character :: Type = ''
    integer :: Ndim
    type(fieldlist), pointer :: next
    real, pointer :: rfield_0d
    real, pointer :: rfield_1d(:)
    real, pointer :: rfield_2d(:,:)
    real, pointer :: rfield_3d(:,:,:)
    real, pointer :: rfield_4d(:,:,:,:)
    real, pointer :: rfield_5d(:,:,:,:,:)
    real, pointer :: rfield_6d(:,:,:,:,:,:)
    real, pointer :: rfield_7d(:,:,:,:,:,:,:)
    double precision, pointer :: dfield_0d
    double precision, pointer :: dfield_1d(:)
    double precision, pointer :: dfield_2d(:,:)
    double precision, pointer :: dfield_3d(:,:,:)
    double precision, pointer :: dfield_4d(:,:,:,:)
    double precision, pointer :: dfield_5d(:,:,:,:,:)
    double precision, pointer :: dfield_6d(:,:,:,:,:,:)
    double precision, pointer :: dfield_7d(:,:,:,:,:,:,:)
    integer, pointer :: ifield_0d
    integer, pointer :: ifield_1d(:)
    integer, pointer :: ifield_2d(:,:)
    integer, pointer :: ifield_3d(:,:,:)
    integer, pointer :: ifield_4d(:,:,:,:)
    integer, pointer :: ifield_5d(:,:,:,:,:)
    integer, pointer :: ifield_6d(:,:,:,:,:,:)
    integer, pointer :: ifield_7d(:,:,:,:,:,:,:)
    logical, pointer :: lfield_0d
    logical, pointer :: lfield_1d(:)
    logical, pointer :: lfield_2d(:,:)
  end type
  type domain
    type(fieldlist), pointer :: head_statevars
  end type
end module

integer(c_int) function fixture_grid() bind(C) result(errors)
  use iso_c_binding
  use module_domain_type
  implicit none
  type(domain) :: grid
  type(fieldlist), target :: head
  type(fieldlist), allocatable, target :: nodes(:)
  type(c_ptr) :: ctx, base
  integer(c_int64_t) :: generation, old_generation
  integer(c_size_t) :: remaining, limit
  integer(c_int) :: status, got
  integer :: n
  real, target :: strided(8), dummy(1,1,1,1)
  interface
    function create(limit) bind(C, name="wt_live_spans_create") result(ctx)
      import
      integer(c_size_t), value :: limit
      type(c_ptr) :: ctx
    end function
    function begin(ctx) bind(C, name="wt_live_spans_begin") result(gen)
      import
      type(c_ptr), value :: ctx
      integer(c_int64_t) :: gen
    end function
    function finish(ctx) bind(C, name="wt_live_spans_finish") result(rc)
      import
      type(c_ptr), value :: ctx
      integer(c_int) :: rc
    end function
    function lookup(ctx, gen, base, typ, width, required, remaining) &
        bind(C, name="wt_live_spans_lookup") result(rc)
      import
      type(c_ptr), value :: ctx, base
      integer(c_int64_t), value :: gen
      integer(c_int), value :: typ
      integer(c_size_t), value :: width, required
      integer(c_size_t) :: remaining
      integer(c_int) :: rc
    end function
    subroutine end_scope(ctx) bind(C, name="wt_live_spans_end")
      import
      type(c_ptr), value :: ctx
    end subroutine
    subroutine destroy(ctx) bind(C, name="wt_live_spans_destroy")
      import
      type(c_ptr), value :: ctx
    end subroutine
  end interface
  errors = 0
  ctx = create(40_c_size_t)
  if (.not. c_associated(ctx)) then
    errors = 1
    return
  end if
  allocate(nodes(27))
  grid%head_statevars => head
  head%next => nodes(1)
  do n = 1, 26
    nodes(n)%next => nodes(n+1)
  end do
  nullify(nodes(27)%next)
  nodes(1)%Type = 'r'
  nodes(1)%Ndim = 0
  allocate(nodes(1)%rfield_0d)
  nodes(2)%Type = 'r'
  nodes(2)%Ndim = 1
  allocate(nodes(2)%rfield_1d(-1:0))
  nodes(3)%Type = 'r'
  nodes(3)%Ndim = 2
  allocate(nodes(3)%rfield_2d(-1:0,-1:0))
  nodes(4)%Type = 'r'
  nodes(4)%Ndim = 3
  allocate(nodes(4)%rfield_3d(-1:0,-1:0,-1:0))
  nodes(5)%Type = 'r'
  nodes(5)%Ndim = 4
  allocate(nodes(5)%rfield_4d(-1:0,-2:-1,1:2,3:5))
  nodes(6)%Type = 'r'
  nodes(6)%Ndim = 5
  allocate(nodes(6)%rfield_5d(-1:0,-1:0,-1:0,-1:0,-1:0))
  nodes(7)%Type = 'r'
  nodes(7)%Ndim = 6
  allocate(nodes(7)%rfield_6d(-1:0,-1:0,-1:0,-1:0,-1:0,-1:0))
  nodes(8)%Type = 'r'
  nodes(8)%Ndim = 7
  allocate(nodes(8)%rfield_7d(-1:0,-1:0,-1:0,-1:0,-1:0,-1:0,-1:0))
  nodes(9)%Type = 'd'
  nodes(9)%Ndim = 0
  allocate(nodes(9)%dfield_0d)
  nodes(10)%Type = 'd'
  nodes(10)%Ndim = 1
  allocate(nodes(10)%dfield_1d(-1:0))
  nodes(11)%Type = 'd'
  nodes(11)%Ndim = 2
  allocate(nodes(11)%dfield_2d(-1:0,-1:0))
  nodes(12)%Type = 'd'
  nodes(12)%Ndim = 3
  allocate(nodes(12)%dfield_3d(-1:0,-1:0,-1:0))
  nodes(13)%Type = 'd'
  nodes(13)%Ndim = 4
  allocate(nodes(13)%dfield_4d(-1:0,-1:0,-1:0,-1:0))
  nodes(14)%Type = 'd'
  nodes(14)%Ndim = 5
  allocate(nodes(14)%dfield_5d(-1:0,-1:0,-1:0,-1:0,-1:0))
  nodes(15)%Type = 'd'
  nodes(15)%Ndim = 6
  allocate(nodes(15)%dfield_6d(-1:0,-1:0,-1:0,-1:0,-1:0,-1:0))
  nodes(16)%Type = 'd'
  nodes(16)%Ndim = 7
  allocate(nodes(16)%dfield_7d(-1:0,-1:0,-1:0,-1:0,-1:0,-1:0,-1:0))
  nodes(17)%Type = 'i'
  nodes(17)%Ndim = 0
  allocate(nodes(17)%ifield_0d)
  nodes(18)%Type = 'i'
  nodes(18)%Ndim = 1
  allocate(nodes(18)%ifield_1d(-1:0))
  nodes(19)%Type = 'i'
  nodes(19)%Ndim = 2
  allocate(nodes(19)%ifield_2d(-1:0,-1:0))
  nodes(20)%Type = 'i'
  nodes(20)%Ndim = 3
  allocate(nodes(20)%ifield_3d(-1:0,-1:0,-1:0))
  nodes(21)%Type = 'i'
  nodes(21)%Ndim = 4
  allocate(nodes(21)%ifield_4d(-1:0,-1:0,-1:0,-1:0))
  nodes(22)%Type = 'i'
  nodes(22)%Ndim = 5
  allocate(nodes(22)%ifield_5d(-1:0,-1:0,-1:0,-1:0,-1:0))
  nodes(23)%Type = 'i'
  nodes(23)%Ndim = 6
  allocate(nodes(23)%ifield_6d(-1:0,-1:0,-1:0,-1:0,-1:0,-1:0))
  nodes(24)%Type = 'i'
  nodes(24)%Ndim = 7
  allocate(nodes(24)%ifield_7d(-1:0,-1:0,-1:0,-1:0,-1:0,-1:0,-1:0))
  nodes(25)%Type = 'l'
  nodes(25)%Ndim = 0
  allocate(nodes(25)%lfield_0d)
  nodes(26)%Type = 'l'
  nodes(26)%Ndim = 1
  allocate(nodes(26)%lfield_1d(-1:0))
  nodes(27)%Type = 'l'
  nodes(27)%Ndim = 2
  allocate(nodes(27)%lfield_2d(-1:0,-1:0))
  generation = begin(ctx)
  limit = 27
  call wt_grid_spans(grid, ctx, limit, status)
  call check(status == 0)
  call check(finish(ctx) == 0)
  base = c_loc(nodes(1)%rfield_0d)
  got = lookup(ctx, generation, base, 1_c_int, 4_c_size_t, 4_c_size_t, remaining)
  call check(got == 0 .and. remaining == 4)
  base = c_loc(nodes(2)%rfield_1d(-1))
  got = lookup(ctx, generation, base, 1_c_int, 4_c_size_t, 8_c_size_t, remaining)
  call check(got == 0 .and. remaining == 8)
  base = c_loc(nodes(3)%rfield_2d(-1,-1))
  got = lookup(ctx, generation, base, 1_c_int, 4_c_size_t, 16_c_size_t, remaining)
  call check(got == 0 .and. remaining == 16)
  base = c_loc(nodes(4)%rfield_3d(-1,-1,-1))
  got = lookup(ctx, generation, base, 1_c_int, 4_c_size_t, 32_c_size_t, remaining)
  call check(got == 0 .and. remaining == 32)
  base = c_loc(nodes(5)%rfield_4d(-1,-2,1,3))
  got = lookup(ctx, generation, base, 1_c_int, 4_c_size_t, 96_c_size_t, remaining)
  call check(got == 0 .and. remaining == 96)
  base = c_loc(nodes(6)%rfield_5d(-1,-1,-1,-1,-1))
  got = lookup(ctx, generation, base, 1_c_int, 4_c_size_t, 128_c_size_t, remaining)
  call check(got == 0 .and. remaining == 128)
  base = c_loc(nodes(7)%rfield_6d(-1,-1,-1,-1,-1,-1))
  got = lookup(ctx, generation, base, 1_c_int, 4_c_size_t, 256_c_size_t, remaining)
  call check(got == 0 .and. remaining == 256)
  base = c_loc(nodes(8)%rfield_7d(-1,-1,-1,-1,-1,-1,-1))
  got = lookup(ctx, generation, base, 1_c_int, 4_c_size_t, 512_c_size_t, remaining)
  call check(got == 0 .and. remaining == 512)
  base = c_loc(nodes(9)%dfield_0d)
  got = lookup(ctx, generation, base, 2_c_int, 8_c_size_t, 8_c_size_t, remaining)
  call check(got == 0 .and. remaining == 8)
  base = c_loc(nodes(10)%dfield_1d(-1))
  got = lookup(ctx, generation, base, 2_c_int, 8_c_size_t, 16_c_size_t, remaining)
  call check(got == 0 .and. remaining == 16)
  base = c_loc(nodes(11)%dfield_2d(-1,-1))
  got = lookup(ctx, generation, base, 2_c_int, 8_c_size_t, 32_c_size_t, remaining)
  call check(got == 0 .and. remaining == 32)
  base = c_loc(nodes(12)%dfield_3d(-1,-1,-1))
  got = lookup(ctx, generation, base, 2_c_int, 8_c_size_t, 64_c_size_t, remaining)
  call check(got == 0 .and. remaining == 64)
  base = c_loc(nodes(13)%dfield_4d(-1,-1,-1,-1))
  got = lookup(ctx, generation, base, 2_c_int, 8_c_size_t, 128_c_size_t, remaining)
  call check(got == 0 .and. remaining == 128)
  base = c_loc(nodes(14)%dfield_5d(-1,-1,-1,-1,-1))
  got = lookup(ctx, generation, base, 2_c_int, 8_c_size_t, 256_c_size_t, remaining)
  call check(got == 0 .and. remaining == 256)
  base = c_loc(nodes(15)%dfield_6d(-1,-1,-1,-1,-1,-1))
  got = lookup(ctx, generation, base, 2_c_int, 8_c_size_t, 512_c_size_t, remaining)
  call check(got == 0 .and. remaining == 512)
  base = c_loc(nodes(16)%dfield_7d(-1,-1,-1,-1,-1,-1,-1))
  got = lookup(ctx, generation, base, 2_c_int, 8_c_size_t, 1024_c_size_t, remaining)
  call check(got == 0 .and. remaining == 1024)
  base = c_loc(nodes(17)%ifield_0d)
  got = lookup(ctx, generation, base, 3_c_int, 4_c_size_t, 4_c_size_t, remaining)
  call check(got == 0 .and. remaining == 4)
  base = c_loc(nodes(18)%ifield_1d(-1))
  got = lookup(ctx, generation, base, 3_c_int, 4_c_size_t, 8_c_size_t, remaining)
  call check(got == 0 .and. remaining == 8)
  base = c_loc(nodes(19)%ifield_2d(-1,-1))
  got = lookup(ctx, generation, base, 3_c_int, 4_c_size_t, 16_c_size_t, remaining)
  call check(got == 0 .and. remaining == 16)
  base = c_loc(nodes(20)%ifield_3d(-1,-1,-1))
  got = lookup(ctx, generation, base, 3_c_int, 4_c_size_t, 32_c_size_t, remaining)
  call check(got == 0 .and. remaining == 32)
  base = c_loc(nodes(21)%ifield_4d(-1,-1,-1,-1))
  got = lookup(ctx, generation, base, 3_c_int, 4_c_size_t, 64_c_size_t, remaining)
  call check(got == 0 .and. remaining == 64)
  base = c_loc(nodes(22)%ifield_5d(-1,-1,-1,-1,-1))
  got = lookup(ctx, generation, base, 3_c_int, 4_c_size_t, 128_c_size_t, remaining)
  call check(got == 0 .and. remaining == 128)
  base = c_loc(nodes(23)%ifield_6d(-1,-1,-1,-1,-1,-1))
  got = lookup(ctx, generation, base, 3_c_int, 4_c_size_t, 256_c_size_t, remaining)
  call check(got == 0 .and. remaining == 256)
  base = c_loc(nodes(24)%ifield_7d(-1,-1,-1,-1,-1,-1,-1))
  got = lookup(ctx, generation, base, 3_c_int, 4_c_size_t, 512_c_size_t, remaining)
  call check(got == 0 .and. remaining == 512)
  base = c_loc(nodes(25)%lfield_0d)
  got = lookup(ctx, generation, base, 4_c_int, 4_c_size_t, 4_c_size_t, remaining)
  call check(got == 0 .and. remaining == 4)
  base = c_loc(nodes(26)%lfield_1d(-1))
  got = lookup(ctx, generation, base, 4_c_int, 4_c_size_t, 8_c_size_t, remaining)
  call check(got == 0 .and. remaining == 8)
  base = c_loc(nodes(27)%lfield_2d(-1,-1))
  got = lookup(ctx, generation, base, 4_c_int, 4_c_size_t, 16_c_size_t, remaining)
  call check(got == 0 .and. remaining == 16)
  base = c_loc(nodes(5)%rfield_4d(-1,-2,1,5))
  got = lookup(ctx, generation, base, 1_c_int, 4_c_size_t, 32_c_size_t, remaining)
  call check(got == 0 .and. remaining == 32)
  got = lookup(ctx, generation, base, 1_c_int, 4_c_size_t, 36_c_size_t, remaining)
  call check(got == 1 .and. remaining == 0)
  base = c_loc(dummy(1,1,1,1))
  got = lookup(ctx, generation, base, 1_c_int, 4_c_size_t, 4_c_size_t, remaining)
  call check(got == 1)
  call end_scope(ctx)
  base = c_loc(nodes(1)%rfield_0d)
  got = lookup(ctx, generation, base, 1_c_int, 4_c_size_t, 4_c_size_t, remaining)
  call check(got == 1)
  old_generation = generation
  generation = begin(ctx)
  limit = 26
  call wt_grid_spans(grid, ctx, limit, status)
  call check(status == 2 .and. finish(ctx) == 2)
  got = lookup(ctx, generation, base, 1_c_int, 4_c_size_t, 4_c_size_t, remaining)
  call check(got == 1)
  nodes(27)%next => nodes(1)
  generation = begin(ctx)
  limit = 40
  call wt_grid_spans(grid, ctx, limit, status)
  call check(status == 2 .and. finish(ctx) == 2)
  nullify(nodes(27)%next)
  generation = begin(ctx)
  deallocate(nodes(2)%rfield_1d)
  nodes(2)%rfield_1d => strided(1:8:2)
  call wt_grid_spans(grid, ctx, limit, status)
  call check(status == 2 .and. finish(ctx) == 2)
  nullify(nodes(2)%rfield_1d)
  allocate(nodes(2)%rfield_1d(0))
  generation = begin(ctx)
  call wt_grid_spans(grid, ctx, limit, status)
  call check(status == 0 .and. finish(ctx) == 0)
  got = lookup(ctx, old_generation, base, 1_c_int, 4_c_size_t, 4_c_size_t, remaining)
  call check(got == 1)
  nullify(grid%head_statevars)
  generation = begin(ctx)
  call wt_grid_spans(grid, ctx, limit, status)
  call check(status == 0 .and. finish(ctx) == 0)
  got = lookup(ctx, generation, base, 1_c_int, 4_c_size_t, 4_c_size_t, remaining)
  call check(got == 1)
  grid%head_statevars => head
  nullify(head%next)
  generation = begin(ctx)
  limit = 0
  call wt_grid_spans(grid, ctx, limit, status)
  call check(status == 0 .and. finish(ctx) == 0)
  head%next => head
  generation = begin(ctx)
  limit = 1
  call wt_grid_spans(grid, ctx, limit, status)
  call check(status == 2 .and. finish(ctx) == 2)
  head%next => nodes(1)
  nullify(nodes(1)%next)
  nodes(1)%Type = 'l'
  nodes(1)%Ndim = 3
  generation = begin(ctx)
  call wt_grid_spans(grid, ctx, limit, status)
  call check(status == 0 .and. finish(ctx) == 0)
  got = lookup(ctx, generation, base, 1_c_int, 4_c_size_t, 4_c_size_t, remaining)
  call check(got == 1)
  nodes(1)%Type = 'r'
  nodes(1)%Ndim = 1
  nullify(nodes(1)%rfield_1d)
  generation = begin(ctx)
  call wt_grid_spans(grid, ctx, limit, status)
  call check(status == 0 .and. finish(ctx) == 0)
  deallocate(nodes(1)%rfield_0d)
  deallocate(nodes(2)%rfield_1d)
  deallocate(nodes(3)%rfield_2d)
  deallocate(nodes(4)%rfield_3d)
  deallocate(nodes(5)%rfield_4d)
  deallocate(nodes(6)%rfield_5d)
  deallocate(nodes(7)%rfield_6d)
  deallocate(nodes(8)%rfield_7d)
  deallocate(nodes(9)%dfield_0d)
  deallocate(nodes(10)%dfield_1d)
  deallocate(nodes(11)%dfield_2d)
  deallocate(nodes(12)%dfield_3d)
  deallocate(nodes(13)%dfield_4d)
  deallocate(nodes(14)%dfield_5d)
  deallocate(nodes(15)%dfield_6d)
  deallocate(nodes(16)%dfield_7d)
  deallocate(nodes(17)%ifield_0d)
  deallocate(nodes(18)%ifield_1d)
  deallocate(nodes(19)%ifield_2d)
  deallocate(nodes(20)%ifield_3d)
  deallocate(nodes(21)%ifield_4d)
  deallocate(nodes(22)%ifield_5d)
  deallocate(nodes(23)%ifield_6d)
  deallocate(nodes(24)%ifield_7d)
  deallocate(nodes(25)%lfield_0d)
  deallocate(nodes(26)%lfield_1d)
  deallocate(nodes(27)%lfield_2d)
  deallocate(nodes)
  call destroy(ctx)
contains
  subroutine check(condition)
    logical, intent(in) :: condition
    if (.not. condition) errors = errors + 1
  end subroutine
end function
