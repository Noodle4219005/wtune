module module_wrf_error
contains
subroutine wrf_message(message)
character(*),intent(in)::message
write(*,*) trim(message)
end subroutine
subroutine wrf_error_fatal3(file,line,message)
character(*),intent(in)::file,message
integer,intent(in)::line
write(*,*) trim(file),line,trim(message)
error stop 1
end subroutine
end module
