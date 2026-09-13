! Small fake-domain fixture: no native WRF module or ABI claim.
module wt_native_capture_fixture
  use iso_c_binding
  use module_domain_type
  implicit none
  type(domain), target, save :: grid
  type(fieldlist), target, save :: head, field
  real, target, save :: values(-1:0,-2:-1,1:2,3:5)
contains
  function native_fixture_grid(tiles) bind(C) result(address)
    integer(c_int), value :: tiles
    type(c_ptr) :: address
    grid%num_tiles = tiles
    grid%id = 1
    grid%head_statevars => head
    head%next => field
    nullify(field%next)
    field%Type = 'r'
    field%Ndim = 4
    field%rfield_4d => values
    address = c_loc(grid)
  end function
  function native_fixture_interior() bind(C) result(address)
    type(c_ptr) :: address
    address = c_loc(values(-1,-2,1,5))
  end function
  subroutine native_fixture_set_id(id) bind(C)
    integer(c_int), value :: id
    grid%id = id
  end subroutine
end module
