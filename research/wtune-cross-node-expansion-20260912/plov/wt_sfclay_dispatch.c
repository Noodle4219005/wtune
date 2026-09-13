#include "wt_sfclay_live.h"

/* Default names match the frozen Intel module ABI. Other compilers must supply
 * the audited linker symbol pair at compile time, and --wrap it at link time.
 * sfclayinit has precisely one native default-LOGICAL address argument. */
#ifndef WT_SFCLAY_INIT_WRAP
#define WT_SFCLAY_INIT_WRAP __wrap_module_sf_sfclay_mp_sfclayinit_
#endif
#ifndef WT_SFCLAY_INIT_REAL
#define WT_SFCLAY_INIT_REAL __real_module_sf_sfclay_mp_sfclayinit_
#endif
extern void WT_SFCLAY_INIT_REAL(int *allowed_to_read);

void WT_SFCLAY_INIT_WRAP(int *allowed_to_read)
{
    wt_sfclay_init_dispatch(allowed_to_read, WT_SFCLAY_INIT_REAL);
}

/* This argument order is the generated 84-address bridge's production ABI. */
void wt_sfclay_bridge_dispatch(void **argv, size_t nargs, void (*real)(void **))
{
    wt_sfclay_live_dispatch(argv, real, nargs);
}
