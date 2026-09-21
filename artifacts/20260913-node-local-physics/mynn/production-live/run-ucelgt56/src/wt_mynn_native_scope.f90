! Ordinary native Fortran ABI: compile only against the pinned WRF modules.
subroutine wt_mynn_native_scope(grid, cfg, id, epoch, status)
  use iso_c_binding
  use module_domain_type, only: domain
  use module_configure, only: grid_config_rec_type
  use module_state_description, only: MYNNPBLSCHEME
  implicit none
  type(domain), intent(in) :: grid
  type(grid_config_rec_type), intent(in) :: cfg
  integer(c_int), intent(out) :: id, epoch, status
  id=0; epoch=0; status=2
  if (storage_size(0.0)/=32 .or. storage_size(0)/=32 .or. storage_size(.false.)/=32) return
  if (cfg%bl_pbl_physics/=MYNNPBLSCHEME) return
  if (grid%id<=0 .or. grid%itimestep<=0) return
  if (grid%id>huge(id) .or. grid%itimestep>huge(epoch)) return
  id=int(grid%id,c_int); epoch=int(grid%itimestep,c_int); status=0
end subroutine
