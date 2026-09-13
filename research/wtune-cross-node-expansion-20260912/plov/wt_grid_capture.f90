! Compile against the frozen WRF module_domain_type.mod with its exact compiler
! and options. This ordinary external uses that compiler's native domain ABI.
subroutine wt_grid_capture(grid, context, max_nodes, num_tiles, status)
  use iso_c_binding
  use module_domain_type, only: domain
  implicit none
  type(domain), intent(in) :: grid
  type(c_ptr), intent(in) :: context
  integer(c_size_t), intent(in) :: max_nodes
  integer(c_int), intent(out) :: num_tiles, status
  interface
    subroutine wt_grid_spans(grid, context, max_nodes, status)
      import domain, c_ptr, c_size_t, c_int
      type(domain), intent(in) :: grid
      type(c_ptr), intent(in) :: context
      integer(c_size_t), intent(in) :: max_nodes
      integer(c_int), intent(out) :: status
    end subroutine
    subroutine fail(context) bind(C, name="wt_live_spans_fail")
      import c_ptr
      type(c_ptr), value :: context
    end subroutine
  end interface
  num_tiles = 0
  status = 2
  ! The KF transaction/schema require four-byte default REAL/INTEGER/LOGICAL.
  ! Check before reading grid members or capturing any descriptor addresses.
  if (storage_size(0.0) /= 32 .or. storage_size(0) /= 32 .or. &
      storage_size(.false.) /= 32) then
    call fail(context)
    return
  end if
  if (grid%num_tiles <= 0 .or. grid%num_tiles > huge(num_tiles)) then
    call fail(context)
    return
  end if
  call wt_grid_spans(grid, context, max_nodes, status)
  if (status /= 0) then
    call fail(context)
    return
  end if
  num_tiles = int(grid%num_tiles, c_int)
end subroutine

! The same module/compiler ABI applies to this identity-only adapter.
subroutine wt_grid_identity(grid, id, status)
  use iso_c_binding
  use module_domain_type, only: domain
  implicit none
  type(domain), intent(in) :: grid
  integer(c_int), intent(out) :: id, status
  id = 0
  status = 2
  if (storage_size(0.0) /= 32 .or. storage_size(0) /= 32 .or. &
      storage_size(.false.) /= 32) return
  if (grid%id <= 0 .or. grid%id > huge(id)) return
  id = int(grid%id, c_int)
  status = 0
end subroutine
