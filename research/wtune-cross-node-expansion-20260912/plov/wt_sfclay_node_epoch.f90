! Native first-RK identity for SFCLAY node-local compatibility. Compile this
! ordinary external against the same frozen module_domain_type.mod/compiler.
subroutine wt_sfclay_grid_epoch(grid, epoch, status)
  use iso_c_binding, only: c_int
  use module_domain_type, only: domain
  implicit none
  type(domain), intent(in) :: grid
  integer(c_int), intent(out) :: epoch, status

  epoch = 0_c_int
  status = 1_c_int
  if (storage_size(0) /= 32 .or. storage_size(0_c_int) /= 32) return
  if (grid%itimestep <= 0 .or. grid%itimestep > huge(epoch)) return
  epoch = int(grid%itimestep, c_int)
  status = 0_c_int
end subroutine wt_sfclay_grid_epoch
