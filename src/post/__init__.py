"""post -- WTune post-processing: inventory, extract, plot, sounding, tc, compare, report
(decision 0056, advisory-only: post-processing and analysis, no autotuning). Every
subcommand wraps an established tool rather than re-implementing it (2026-09-29 update):
diffwrf (compare), cdo/ncdump (inventory), wrf-python/cartopy (extract/plot/sounding),
TempestExtremes/MET tc_pairs/tc_stat/tc_rmw (tc). post/tools.py locates and runs all of
them (WRF_TREE, PATH, WTUNE_POSTPROC_VIEW), raising a clear error naming the missing tool
and bin/install-postproc.sh instead of silently falling back to a self-written substitute.
Heavy imports (wrf-python/matplotlib/metpy/cartopy/netCDF4) are all lazy, inside each
subcommand's own function, so `wtune post` subcommands that do not need them (inventory,
compare) never require them installed, and `wtune --help` never imports any of it.

See post/cli.py for the `wtune post <sub>` subcommand list, post/atcf.py for the ATCF
a-deck/b-deck reader/writer, post/met_tc.py for the MET wrapper, and post/tc_metrics.py for
the one surviving self-written metric (quadrant wind radii) and why it stays that way.
"""
