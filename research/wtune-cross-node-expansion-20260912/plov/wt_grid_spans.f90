! Read-only shim: build with the exact native WRF compiler/module/kind ABI.
! Only the Type/Ndim-selected pointer may be inspected: other pointers in
! WRF fieldlist are not default initialized. The head node is a sentinel.
! Caller owns begin/finish/end and excludes frees/reallocation/concurrency.
subroutine wt_grid_spans(grid, context, max_nodes, status)
  use iso_c_binding
  use module_domain_type, only: domain, fieldlist
  implicit none
  type(domain), intent(in) :: grid
  type(c_ptr), intent(in) :: context
  integer(c_size_t), intent(in) :: max_nodes
  integer(c_int), intent(out) :: status
  type(fieldlist), pointer :: node
  type(c_ptr) :: base
  integer(c_size_t) :: visited, bytes, width
  integer(c_int) :: typ
  interface
    function add_span(ctx, ptr, cap, typ, width, rank) bind(C, name="wt_live_spans_add") result(rc)
      import
      type(c_ptr), value :: ctx, ptr
      integer(c_size_t), value :: cap, width
      integer(c_int), value :: typ, rank
      integer(c_int) :: rc
    end function
    subroutine fail(ctx) bind(C, name="wt_live_spans_fail")
      import
      type(c_ptr), value :: ctx
    end subroutine
  end interface
  status = 2
  if (.not. c_associated(context) .or. max_nodes < 0) then
    call fail(context)
    return
  end if
  if (.not. associated(grid%head_statevars)) then
    status = 0
    return
  end if
  node => grid%head_statevars
  visited = 0
  do
    if (.not. associated(node%next)) then
      status = 0
      return
    end if
    if (visited >= max_nodes) exit
    node => node%next
    if (associated(node, grid%head_statevars)) exit
    visited = visited + 1
    select case (node%Type)
    case ('r')
      typ = 1
      select case (node%Ndim)
      case (0)
        if (.not. associated(node%rfield_0d)) cycle
        width = storage_size(node%rfield_0d, kind=c_size_t) / 8
        bytes = width
        base = c_loc(node%rfield_0d)
      case (1)
        if (.not. associated(node%rfield_1d)) cycle
        if (.not. is_contiguous(node%rfield_1d)) exit
        width = storage_size(node%rfield_1d, kind=c_size_t) / 8
        bytes = span_bytes(shape(node%rfield_1d, kind=c_size_t), width)
        if (bytes < 0) exit
        if (bytes == 0) cycle
        if (bytes / width /= size(node%rfield_1d, kind=c_size_t)) exit
        base = c_loc(node%rfield_1d(lbound(node%rfield_1d,1)))
      case (2)
        if (.not. associated(node%rfield_2d)) cycle
        if (.not. is_contiguous(node%rfield_2d)) exit
        width = storage_size(node%rfield_2d, kind=c_size_t) / 8
        bytes = span_bytes(shape(node%rfield_2d, kind=c_size_t), width)
        if (bytes < 0) exit
        if (bytes == 0) cycle
        if (bytes / width /= size(node%rfield_2d, kind=c_size_t)) exit
        base = c_loc(node%rfield_2d(lbound(node%rfield_2d,1),lbound(node%rfield_2d,2)))
      case (3)
        if (.not. associated(node%rfield_3d)) cycle
        if (.not. is_contiguous(node%rfield_3d)) exit
        width = storage_size(node%rfield_3d, kind=c_size_t) / 8
        bytes = span_bytes(shape(node%rfield_3d, kind=c_size_t), width)
        if (bytes < 0) exit
        if (bytes == 0) cycle
        if (bytes / width /= size(node%rfield_3d, kind=c_size_t)) exit
        base = c_loc(node%rfield_3d( &
          lbound(node%rfield_3d,1),lbound(node%rfield_3d,2),lbound(node%rfield_3d,3)))
      case (4)
        if (.not. associated(node%rfield_4d)) cycle
        if (.not. is_contiguous(node%rfield_4d)) exit
        width = storage_size(node%rfield_4d, kind=c_size_t) / 8
        bytes = span_bytes(shape(node%rfield_4d, kind=c_size_t), width)
        if (bytes < 0) exit
        if (bytes == 0) cycle
        if (bytes / width /= size(node%rfield_4d, kind=c_size_t)) exit
        base = c_loc(node%rfield_4d( &
          lbound(node%rfield_4d,1),lbound(node%rfield_4d,2),lbound(node%rfield_4d,3), &
          lbound(node%rfield_4d,4)))
      case (5)
        if (.not. associated(node%rfield_5d)) cycle
        if (.not. is_contiguous(node%rfield_5d)) exit
        width = storage_size(node%rfield_5d, kind=c_size_t) / 8
        bytes = span_bytes(shape(node%rfield_5d, kind=c_size_t), width)
        if (bytes < 0) exit
        if (bytes == 0) cycle
        if (bytes / width /= size(node%rfield_5d, kind=c_size_t)) exit
        base = c_loc(node%rfield_5d( &
          lbound(node%rfield_5d,1),lbound(node%rfield_5d,2),lbound(node%rfield_5d,3), &
          lbound(node%rfield_5d,4),lbound(node%rfield_5d,5)))
      case (6)
        if (.not. associated(node%rfield_6d)) cycle
        if (.not. is_contiguous(node%rfield_6d)) exit
        width = storage_size(node%rfield_6d, kind=c_size_t) / 8
        bytes = span_bytes(shape(node%rfield_6d, kind=c_size_t), width)
        if (bytes < 0) exit
        if (bytes == 0) cycle
        if (bytes / width /= size(node%rfield_6d, kind=c_size_t)) exit
        base = c_loc(node%rfield_6d( &
          lbound(node%rfield_6d,1),lbound(node%rfield_6d,2),lbound(node%rfield_6d,3), &
          lbound(node%rfield_6d,4),lbound(node%rfield_6d,5),lbound(node%rfield_6d,6)))
      case (7)
        if (.not. associated(node%rfield_7d)) cycle
        if (.not. is_contiguous(node%rfield_7d)) exit
        width = storage_size(node%rfield_7d, kind=c_size_t) / 8
        bytes = span_bytes(shape(node%rfield_7d, kind=c_size_t), width)
        if (bytes < 0) exit
        if (bytes == 0) cycle
        if (bytes / width /= size(node%rfield_7d, kind=c_size_t)) exit
        base = c_loc(node%rfield_7d( &
          lbound(node%rfield_7d,1),lbound(node%rfield_7d,2),lbound(node%rfield_7d,3), &
          lbound(node%rfield_7d,4),lbound(node%rfield_7d,5),lbound(node%rfield_7d,6), &
          lbound(node%rfield_7d,7)))
      case default
        cycle
      end select
    case ('d')
      typ = 2
      select case (node%Ndim)
      case (0)
        if (.not. associated(node%dfield_0d)) cycle
        width = storage_size(node%dfield_0d, kind=c_size_t) / 8
        bytes = width
        base = c_loc(node%dfield_0d)
      case (1)
        if (.not. associated(node%dfield_1d)) cycle
        if (.not. is_contiguous(node%dfield_1d)) exit
        width = storage_size(node%dfield_1d, kind=c_size_t) / 8
        bytes = span_bytes(shape(node%dfield_1d, kind=c_size_t), width)
        if (bytes < 0) exit
        if (bytes == 0) cycle
        if (bytes / width /= size(node%dfield_1d, kind=c_size_t)) exit
        base = c_loc(node%dfield_1d(lbound(node%dfield_1d,1)))
      case (2)
        if (.not. associated(node%dfield_2d)) cycle
        if (.not. is_contiguous(node%dfield_2d)) exit
        width = storage_size(node%dfield_2d, kind=c_size_t) / 8
        bytes = span_bytes(shape(node%dfield_2d, kind=c_size_t), width)
        if (bytes < 0) exit
        if (bytes == 0) cycle
        if (bytes / width /= size(node%dfield_2d, kind=c_size_t)) exit
        base = c_loc(node%dfield_2d(lbound(node%dfield_2d,1),lbound(node%dfield_2d,2)))
      case (3)
        if (.not. associated(node%dfield_3d)) cycle
        if (.not. is_contiguous(node%dfield_3d)) exit
        width = storage_size(node%dfield_3d, kind=c_size_t) / 8
        bytes = span_bytes(shape(node%dfield_3d, kind=c_size_t), width)
        if (bytes < 0) exit
        if (bytes == 0) cycle
        if (bytes / width /= size(node%dfield_3d, kind=c_size_t)) exit
        base = c_loc(node%dfield_3d( &
          lbound(node%dfield_3d,1),lbound(node%dfield_3d,2),lbound(node%dfield_3d,3)))
      case (4)
        if (.not. associated(node%dfield_4d)) cycle
        if (.not. is_contiguous(node%dfield_4d)) exit
        width = storage_size(node%dfield_4d, kind=c_size_t) / 8
        bytes = span_bytes(shape(node%dfield_4d, kind=c_size_t), width)
        if (bytes < 0) exit
        if (bytes == 0) cycle
        if (bytes / width /= size(node%dfield_4d, kind=c_size_t)) exit
        base = c_loc(node%dfield_4d( &
          lbound(node%dfield_4d,1),lbound(node%dfield_4d,2),lbound(node%dfield_4d,3), &
          lbound(node%dfield_4d,4)))
      case (5)
        if (.not. associated(node%dfield_5d)) cycle
        if (.not. is_contiguous(node%dfield_5d)) exit
        width = storage_size(node%dfield_5d, kind=c_size_t) / 8
        bytes = span_bytes(shape(node%dfield_5d, kind=c_size_t), width)
        if (bytes < 0) exit
        if (bytes == 0) cycle
        if (bytes / width /= size(node%dfield_5d, kind=c_size_t)) exit
        base = c_loc(node%dfield_5d( &
          lbound(node%dfield_5d,1),lbound(node%dfield_5d,2),lbound(node%dfield_5d,3), &
          lbound(node%dfield_5d,4),lbound(node%dfield_5d,5)))
      case (6)
        if (.not. associated(node%dfield_6d)) cycle
        if (.not. is_contiguous(node%dfield_6d)) exit
        width = storage_size(node%dfield_6d, kind=c_size_t) / 8
        bytes = span_bytes(shape(node%dfield_6d, kind=c_size_t), width)
        if (bytes < 0) exit
        if (bytes == 0) cycle
        if (bytes / width /= size(node%dfield_6d, kind=c_size_t)) exit
        base = c_loc(node%dfield_6d( &
          lbound(node%dfield_6d,1),lbound(node%dfield_6d,2),lbound(node%dfield_6d,3), &
          lbound(node%dfield_6d,4),lbound(node%dfield_6d,5),lbound(node%dfield_6d,6)))
      case (7)
        if (.not. associated(node%dfield_7d)) cycle
        if (.not. is_contiguous(node%dfield_7d)) exit
        width = storage_size(node%dfield_7d, kind=c_size_t) / 8
        bytes = span_bytes(shape(node%dfield_7d, kind=c_size_t), width)
        if (bytes < 0) exit
        if (bytes == 0) cycle
        if (bytes / width /= size(node%dfield_7d, kind=c_size_t)) exit
        base = c_loc(node%dfield_7d( &
          lbound(node%dfield_7d,1),lbound(node%dfield_7d,2),lbound(node%dfield_7d,3), &
          lbound(node%dfield_7d,4),lbound(node%dfield_7d,5),lbound(node%dfield_7d,6), &
          lbound(node%dfield_7d,7)))
      case default
        cycle
      end select
    case ('i')
      typ = 3
      select case (node%Ndim)
      case (0)
        if (.not. associated(node%ifield_0d)) cycle
        width = storage_size(node%ifield_0d, kind=c_size_t) / 8
        bytes = width
        base = c_loc(node%ifield_0d)
      case (1)
        if (.not. associated(node%ifield_1d)) cycle
        if (.not. is_contiguous(node%ifield_1d)) exit
        width = storage_size(node%ifield_1d, kind=c_size_t) / 8
        bytes = span_bytes(shape(node%ifield_1d, kind=c_size_t), width)
        if (bytes < 0) exit
        if (bytes == 0) cycle
        if (bytes / width /= size(node%ifield_1d, kind=c_size_t)) exit
        base = c_loc(node%ifield_1d(lbound(node%ifield_1d,1)))
      case (2)
        if (.not. associated(node%ifield_2d)) cycle
        if (.not. is_contiguous(node%ifield_2d)) exit
        width = storage_size(node%ifield_2d, kind=c_size_t) / 8
        bytes = span_bytes(shape(node%ifield_2d, kind=c_size_t), width)
        if (bytes < 0) exit
        if (bytes == 0) cycle
        if (bytes / width /= size(node%ifield_2d, kind=c_size_t)) exit
        base = c_loc(node%ifield_2d(lbound(node%ifield_2d,1),lbound(node%ifield_2d,2)))
      case (3)
        if (.not. associated(node%ifield_3d)) cycle
        if (.not. is_contiguous(node%ifield_3d)) exit
        width = storage_size(node%ifield_3d, kind=c_size_t) / 8
        bytes = span_bytes(shape(node%ifield_3d, kind=c_size_t), width)
        if (bytes < 0) exit
        if (bytes == 0) cycle
        if (bytes / width /= size(node%ifield_3d, kind=c_size_t)) exit
        base = c_loc(node%ifield_3d( &
          lbound(node%ifield_3d,1),lbound(node%ifield_3d,2),lbound(node%ifield_3d,3)))
      case (4)
        if (.not. associated(node%ifield_4d)) cycle
        if (.not. is_contiguous(node%ifield_4d)) exit
        width = storage_size(node%ifield_4d, kind=c_size_t) / 8
        bytes = span_bytes(shape(node%ifield_4d, kind=c_size_t), width)
        if (bytes < 0) exit
        if (bytes == 0) cycle
        if (bytes / width /= size(node%ifield_4d, kind=c_size_t)) exit
        base = c_loc(node%ifield_4d( &
          lbound(node%ifield_4d,1),lbound(node%ifield_4d,2),lbound(node%ifield_4d,3), &
          lbound(node%ifield_4d,4)))
      case (5)
        if (.not. associated(node%ifield_5d)) cycle
        if (.not. is_contiguous(node%ifield_5d)) exit
        width = storage_size(node%ifield_5d, kind=c_size_t) / 8
        bytes = span_bytes(shape(node%ifield_5d, kind=c_size_t), width)
        if (bytes < 0) exit
        if (bytes == 0) cycle
        if (bytes / width /= size(node%ifield_5d, kind=c_size_t)) exit
        base = c_loc(node%ifield_5d( &
          lbound(node%ifield_5d,1),lbound(node%ifield_5d,2),lbound(node%ifield_5d,3), &
          lbound(node%ifield_5d,4),lbound(node%ifield_5d,5)))
      case (6)
        if (.not. associated(node%ifield_6d)) cycle
        if (.not. is_contiguous(node%ifield_6d)) exit
        width = storage_size(node%ifield_6d, kind=c_size_t) / 8
        bytes = span_bytes(shape(node%ifield_6d, kind=c_size_t), width)
        if (bytes < 0) exit
        if (bytes == 0) cycle
        if (bytes / width /= size(node%ifield_6d, kind=c_size_t)) exit
        base = c_loc(node%ifield_6d( &
          lbound(node%ifield_6d,1),lbound(node%ifield_6d,2),lbound(node%ifield_6d,3), &
          lbound(node%ifield_6d,4),lbound(node%ifield_6d,5),lbound(node%ifield_6d,6)))
      case (7)
        if (.not. associated(node%ifield_7d)) cycle
        if (.not. is_contiguous(node%ifield_7d)) exit
        width = storage_size(node%ifield_7d, kind=c_size_t) / 8
        bytes = span_bytes(shape(node%ifield_7d, kind=c_size_t), width)
        if (bytes < 0) exit
        if (bytes == 0) cycle
        if (bytes / width /= size(node%ifield_7d, kind=c_size_t)) exit
        base = c_loc(node%ifield_7d( &
          lbound(node%ifield_7d,1),lbound(node%ifield_7d,2),lbound(node%ifield_7d,3), &
          lbound(node%ifield_7d,4),lbound(node%ifield_7d,5),lbound(node%ifield_7d,6), &
          lbound(node%ifield_7d,7)))
      case default
        cycle
      end select
    case ('l')
      typ = 4
      select case (node%Ndim)
      case (0)
        if (.not. associated(node%lfield_0d)) cycle
        width = storage_size(node%lfield_0d, kind=c_size_t) / 8
        bytes = width
        base = c_loc(node%lfield_0d)
      case (1)
        if (.not. associated(node%lfield_1d)) cycle
        if (.not. is_contiguous(node%lfield_1d)) exit
        width = storage_size(node%lfield_1d, kind=c_size_t) / 8
        bytes = span_bytes(shape(node%lfield_1d, kind=c_size_t), width)
        if (bytes < 0) exit
        if (bytes == 0) cycle
        if (bytes / width /= size(node%lfield_1d, kind=c_size_t)) exit
        base = c_loc(node%lfield_1d(lbound(node%lfield_1d,1)))
      case (2)
        if (.not. associated(node%lfield_2d)) cycle
        if (.not. is_contiguous(node%lfield_2d)) exit
        width = storage_size(node%lfield_2d, kind=c_size_t) / 8
        bytes = span_bytes(shape(node%lfield_2d, kind=c_size_t), width)
        if (bytes < 0) exit
        if (bytes == 0) cycle
        if (bytes / width /= size(node%lfield_2d, kind=c_size_t)) exit
        base = c_loc(node%lfield_2d(lbound(node%lfield_2d,1),lbound(node%lfield_2d,2)))
      case default
        cycle
      end select
    case default
      cycle
    end select
    status = add_span(context, base, bytes, typ, width, int(node%Ndim,c_int))
    if (status /= 0) exit
  end do
  call fail(context)
  status = 2
contains
  function span_bytes(extents, width) result(bytes)
    integer(c_size_t), intent(in) :: extents(:), width
    integer(c_size_t) :: bytes
    integer :: d
    bytes = -1
    if (width <= 0 .or. any(extents < 0)) return
    if (any(extents == 0)) then
      bytes = 0
      return
    end if
    bytes = width
    do d = 1, size(extents)
      if (bytes > huge(bytes) / extents(d)) then
        bytes = -1
        return
      end if
      bytes = bytes * extents(d)
    end do
  end function
end subroutine
