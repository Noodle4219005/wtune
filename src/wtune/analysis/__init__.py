"""analysis -- standard-tool TC verification, wrapping wrf-python + TempestExtremes.

Replaces the ad-hoc storm-centre logic in predictor/verify_obs.py's storm_centres()
(global ocean argmin of SLP, no track continuity; domain-max wind, not eyewall wind)
with:

  analysis.diagnostics  wrf-python (wrf.getvar) derived CF-NetCDF fields (slp, uvmet10,
                        wspd10, pw, td2) suitable as TempestExtremes input.
  analysis.tc_track     TempestExtremes DetectNodes + StitchNodes tracking, with a
                        side-by-side comparison against predictor/verify_obs.py's naive
                        method and the HURDAT2 best track.
  analysis.cli          `wtune/cli.py analyze` subcommand: tc-track, diag.

See analysis/diagnostics.py and analysis/tc_track.py module docstrings for the exact
TempestExtremes CLI parameters and why each was chosen.
"""
