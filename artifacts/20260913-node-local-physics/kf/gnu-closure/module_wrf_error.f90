module module_wrf_error
use iso_c_binding
integer(c_int), bind(C,name="kf_diag_messages") :: messages=0
integer(c_int), bind(C,name="kf_diag_fatals") :: fatals=0
contains
subroutine wrf_message(message)
character(*), intent(in) :: message
messages=messages+1
write(*,*) trim(message)
end subroutine
subroutine wrf_error_fatal3(file,line,message)
character(*), intent(in) :: file,message
integer, intent(in) :: line
fatals=fatals+1
write(*,*) trim(file),line,trim(message)
error stop 1
end subroutine
end module
