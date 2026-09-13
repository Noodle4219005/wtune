! Ordinary Fortran ABI: native grid/config references, then five c_int references.
! Requires the matching WRF compiler, module files and default-kind options.
subroutine wt_sfclay_native_query(grid, cfg, grid_id, actual_scheme, &
                                  actual_fractional, expected_scheme, status)
  use iso_c_binding, only: c_int
  use module_domain_type, only: domain
  use module_configure, only: grid_config_rec_type
  use module_driver_constants, only: max_domains
  use module_state_description, only: SFCLAYSCHEME
  implicit none
  type(domain), intent(in) :: grid
  type(grid_config_rec_type), intent(in) :: cfg
  integer(c_int), intent(out) :: grid_id, actual_scheme, actual_fractional
  integer(c_int), intent(out) :: expected_scheme, status
  integer :: native_id, getter_scheme, getter_fractional
  ! Status numbers mirror wt_sfclay_native_status in the C header.
  integer(c_int), parameter :: kind_error = 2, id_error = 3, mismatch = 4
  integer(c_int), parameter :: scheme_error = 5, fractional_error = 6
  interface
    subroutine nl_get_sf_sfclay_physics(id, value)
      integer, intent(in) :: id
      integer, intent(out) :: value
    end subroutine
    subroutine nl_get_fractional_seaice(id, value)
      integer, intent(in) :: id
      integer, intent(out) :: value
    end subroutine
  end interface

  grid_id = 0_c_int
  actual_scheme = 0_c_int
  actual_fractional = 0_c_int
  expected_scheme = 0_c_int
  status = kind_error
  ! Do not touch either native derived type or call a getter before these checks.
  if (storage_size(0.0) /= 32 .or. storage_size(0) /= 32 .or. &
      storage_size(.false.) /= 32 .or. storage_size(0_c_int) /= 32) return

  expected_scheme = int(SFCLAYSCHEME, c_int)
  native_id = grid%id
  grid_id = int(native_id, c_int)
  status = id_error
  if (native_id < 1 .or. native_id > max_domains) return

  actual_scheme = int(cfg%sf_sfclay_physics, c_int)
  actual_fractional = int(cfg%fractional_seaice, c_int)
  call nl_get_sf_sfclay_physics(native_id, getter_scheme)
  call nl_get_fractional_seaice(native_id, getter_fractional)
  status = mismatch
  if (cfg%sf_sfclay_physics /= getter_scheme) return
  if (cfg%fractional_seaice /= getter_fractional) return
  status = scheme_error
  if (cfg%sf_sfclay_physics /= SFCLAYSCHEME) return
  status = fractional_error
  if (cfg%fractional_seaice /= 0) return
  status = 0_c_int
end subroutine
