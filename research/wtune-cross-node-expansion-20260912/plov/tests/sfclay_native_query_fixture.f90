! Small synthetic modules exposing only fields used by the query, not a WRF ABI model.
module module_driver_constants
  implicit none
  integer, parameter :: max_domains = 5
end module

module module_state_description
  implicit none
  integer, parameter :: SFCLAYSCHEME = MOCK_SCHEME
end module

module module_domain_type
  implicit none
  type domain
    integer :: padding(3) = 77
    integer :: id = 1
  end type
end module

module module_configure
  use module_driver_constants, only: max_domains
  implicit none
  type grid_config_rec_type
    real :: padding(2) = 3.0
    integer :: sf_sfclay_physics = 0
    integer :: fractional_seaice = 0
  end type
  type model_config_rec_type
    integer :: sf_sfclay_physics(max_domains) = 0
    integer :: fractional_seaice = 0
  end type
  type(model_config_rec_type), save :: model_config_rec
end module

module sfclay_query_fixture
  use iso_c_binding
  use module_domain_type
  use module_configure
  implicit none
  type(domain), target, save :: grid
  type(grid_config_rec_type), target, save :: cfg
  integer, save :: getter_calls = 0
contains
  subroutine fixture_configure(id, scheme, frac, model_scheme, model_frac, gp, cp) bind(C)
    integer(c_int), value :: id, scheme, frac, model_scheme, model_frac
    type(c_ptr), intent(out) :: gp, cp
    grid%id = id
    cfg%sf_sfclay_physics = scheme
    cfg%fractional_seaice = frac
    model_config_rec%sf_sfclay_physics = model_scheme
    model_config_rec%fractional_seaice = model_frac
    getter_calls = 0
    gp = c_loc(grid)
    cp = c_loc(cfg)
  end subroutine

  function fixture_getter_calls() bind(C) result(count)
    integer(c_int) :: count
    count = int(getter_calls, c_int)
  end function
end module

! The native getters are ordinary Fortran with two default-INTEGER references.
subroutine nl_get_sf_sfclay_physics(id, value)
  use module_configure, only: model_config_rec
  use sfclay_query_fixture, only: getter_calls
  implicit none
  integer, intent(in) :: id
  integer, intent(out) :: value
  getter_calls = getter_calls + 1
  value = model_config_rec%sf_sfclay_physics(id)
end subroutine

subroutine nl_get_fractional_seaice(id, value)
  use module_driver_constants, only: max_domains
  use module_configure, only: model_config_rec
  use sfclay_query_fixture, only: getter_calls
  implicit none
  integer, intent(in) :: id
  integer, intent(out) :: value
  if (id < 1 .or. id > max_domains) error stop 'invalid id reached getter'
  getter_calls = getter_calls + 1
  value = model_config_rec%fractional_seaice
end subroutine
