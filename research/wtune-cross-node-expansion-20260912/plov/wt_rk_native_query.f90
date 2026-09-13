! Ordinary Fortran ABI: native config reference, c_int rk_step,
! then a c_int output array.  Build only with matching WRF module/default kinds.
subroutine wt_rk_native_query(cfg, rk_step, values)
  use iso_c_binding, only: c_int
  use module_configure, only: grid_config_rec_type
  implicit none
  type(grid_config_rec_type), intent(in) :: cfg
  integer(c_int), intent(in) :: rk_step
  integer(c_int), intent(out) :: values(17)
  integer :: native_time_step
  integer(c_int), parameter :: null_error = 1_c_int, kind_error = 2_c_int
  integer(c_int), parameter :: step_error = 3_c_int
  interface
    subroutine nl_get_time_step(id, value)
      integer, intent(in) :: id
      integer, intent(out) :: value
    end subroutine nl_get_time_step
  end interface

  values = 0_c_int
  values(1) = null_error
  ! No cfg member, rk_step, or getter is touched before the native-kind gate.
  if (storage_size(0) /= 32 .or. storage_size(.false.) /= 32 .or. &
      storage_size(0.0) /= 32 .or. storage_size(0_c_int) /= 32) then
    values(1) = kind_error
    return
  end if
  if (rk_step <= 0) then
    values(1) = step_error
    return
  end if

  values(2) = int(cfg%rk_ord, c_int)
  values(3) = merge(1_c_int, 0_c_int, cfg%zadvect_implicit)
  values(4) = int(cfg%h_sca_adv_order, c_int)
  values(5) = int(cfg%v_sca_adv_order, c_int)
  values(6) = int(cfg%cu_physics, c_int)
  values(7) = int(cfg%diff_opt, c_int)
  values(8) = int(cfg%bl_pbl_physics, c_int)
  ! The RK caller gets time_step through this native getter with id 1.
  call nl_get_time_step(1, native_time_step)
  values(9) = int(native_time_step, c_int)
  values(10) = merge(1_c_int, 0_c_int, cfg%specified)
  values(11) = merge(1_c_int, 0_c_int, cfg%nested)
  values(12) = merge(1_c_int, 0_c_int, cfg%periodic_x)
  values(13) = merge(1_c_int, 0_c_int, cfg%polar)
  values(14) = merge(1_c_int, 0_c_int, cfg%open_xs)
  values(15) = merge(1_c_int, 0_c_int, cfg%open_xe)
  values(16) = merge(1_c_int, 0_c_int, cfg%open_ys)
  values(17) = merge(1_c_int, 0_c_int, cfg%open_ye)
  values(1) = 0_c_int
end subroutine wt_rk_native_query
