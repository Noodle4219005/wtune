#include "wt_sfclay_scope.h"
#include "wt_sfclay_live.h"

static _Thread_local uint64_t surface_token;

void wt_sfclay_surface_hook_begin(void)
{
    /* The live runtime rejects reentry before this one-slot token is changed. */
    surface_token = wt_sfclay_surface_begin();
}

void wt_sfclay_surface_hook_end(void)
{
    uint64_t token = surface_token;
    surface_token = 0;
    /* Clear the hook slot before runtime end can restore cancellation. */
    wt_sfclay_surface_end(token);
}
