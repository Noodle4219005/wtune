! Small ordinary-Fortran fixture; it intentionally does not model a WRF ABI.
module module_configure
  implicit none
  type grid_config_rec_type
    real :: padding = 3.0
    integer :: rk_ord = 0
    logical :: zadvect_implicit = .false.
    integer :: h_sca_adv_order = 0, v_sca_adv_order = 0
    integer :: cu_physics = 0, diff_opt = 0, bl_pbl_physics = 0, time_step = 0
    logical :: specified = .false., nested = .false., periodic_x = .false., polar = .false.
    logical :: open_xs = .false., open_xe = .false., open_ys = .false., open_ye = .false.
  end type
  type model_config_rec_type
    integer :: time_step = 0
  end type
  type(model_config_rec_type), save :: model_config_rec
end module

module rk_query_fixture
  use iso_c_binding
  use module_configure
  implicit none
  type(grid_config_rec_type), target, save :: cfg
  integer, save :: getter_calls = 0
contains
  subroutine fixture_configure(values, cp) bind(C)
    integer(c_int), intent(in) :: values(16)
    type(c_ptr), intent(out) :: cp
    cfg%rk_ord = values(1)
    cfg%zadvect_implicit = values(2) /= 0
    cfg%h_sca_adv_order = values(3); cfg%v_sca_adv_order = values(4)
    cfg%cu_physics = values(5); cfg%diff_opt = values(6); cfg%bl_pbl_physics = values(7)
    cfg%time_step = values(8)
    cfg%specified = values(9) /= 0; cfg%nested = values(10) /= 0
    cfg%periodic_x = values(11) /= 0; cfg%polar = values(12) /= 0
    cfg%open_xs = values(13) /= 0; cfg%open_xe = values(14) /= 0
    cfg%open_ys = values(15) /= 0; cfg%open_ye = values(16) /= 0
    getter_calls = 0
    cp = c_loc(cfg)
  end subroutine

  subroutine fixture_set_global_time_step(value) bind(C)
    integer(c_int), value :: value
    model_config_rec%time_step = value
  end subroutine

  function fixture_getter_calls() bind(C) result(count)
    integer(c_int) :: count
    count = int(getter_calls, c_int)
  end function
end module

subroutine nl_get_time_step(id, value)
  use module_configure, only: model_config_rec
  use rk_query_fixture, only: getter_calls
  implicit none
  integer, intent(in) :: id
  integer, intent(out) :: value
  if (id /= 1) error stop 'rk query must use native id 1'
  getter_calls = getter_calls + 1
  value = model_config_rec%time_step
end subroutine
