module module_domain_type
use iso_c_binding
type, bind(C) :: domain
integer(c_int) :: id,itimestep
end type
end module
module module_configure
use iso_c_binding
type, bind(C) :: grid_config_rec_type
integer(c_int) :: bl_pbl_physics
end type
end module
module module_state_description
integer,parameter :: MYNNPBLSCHEME=5
end module
