#ifndef WT_SFCLAY_SCOPE_H
#define WT_SFCLAY_SCOPE_H
/* Pair around the surface driver on the same thread. Direct surface calls get
 * a disabled live activation; nested or unmatched hooks are structural fatal.
 * These hooks do not install a native wrapper or modify wtimer integration. */
void wt_sfclay_surface_hook_begin(void);
void wt_sfclay_surface_hook_end(void);
#endif
