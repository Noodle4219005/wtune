# SFCLAY native adapter/bootstrap draft — 2026-09-13

Added independent SFCLAY native capture/configuration and explicit bootstrap
files. The adapter calls ordinary `wt_grid_capture_`/`wt_grid_identity_`, uses
`omp_in_parallel`, and gives `wt_sfclay_live_configure` the existing native
query callback. It does not decode grid/config data. Selector `0` accepts all,
positive selectors require the real identity callback, and negative selectors
are rejected before configuration.

All three build proof flags are strict booleans. Unattested bootstrap defaults
to refusal and makes no native callback; environment variables only parse
positive decimal budgets. Configuration is one-shot; a failed live install
returns the native adapter to its unconfigured state.

GCC 16.2 C11 strict test command (without sanitizer flags):

```text
pixi run python -m unittest plov.tests.test_sfclay_native plov.tests.test_sfclay_bootstrap -v
```

Result: 2 narrow tests passed in 0.206 s. The C fixture uses a mock live
configure function and ordinary fake-Fortran capture/identity callbacks; it
does not establish live fallback behavior. The bootstrap test only inspected
source text, so it does not validate bootstrap compilation, CSV output,
environment/budget rejection, counters, or proof state. No WRF/Intel/native
ABI execution or production enablement ran. This receipt is rejected pending a
full independent rewrite and validation.
