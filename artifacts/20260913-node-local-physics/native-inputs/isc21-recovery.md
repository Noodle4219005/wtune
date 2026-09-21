# ISC21 retained archive recovery inventory (header-only)

`tarfile.open(..., "r:")` inspected seekable headers only; no payload was read, hashed, or extracted. Prefix is `WRF_ISC21_3_DOMAIN/`.

| member | bytes | data offset |
|---|---:|---:|
| wrfbdy_d01 | 964,952,328 | 1,024 |
| qr_acr_qsV2.dat | 43,764,288 | 2,820,771,840 |
| freezeH2O.dat | 254,944,848 | 3,817,735,680 |
| qr_acr_qgV2.dat | 89,959,776 | 4,072,681,472 |
| namelist.input | 4,179 | 5,624,946,176 |
| wrfinput_d01 | 1,087,552,017 | 5,624,951,296 |

Retained `prep_d01_1071708/namelist.input` is BP5 (`io_form_input/boundary/history=14`, `.bp` names), 400 steps (`run_minutes=100`, dt=15), nproc 7×32, tiles 12. The original frozen archive members are NetCDF-style names with no `.bp`; restoring those small-listed members supports a separate-output **form-2** recovery harness directly. Preserving the retained BP I/O instead requires the existing `nc2bp.sbatch` converter: `wtune/tools/wrf_nc2bp.py`, source prefix `WRF_ISC21_3_DOMAIN`, output `atlas/isc21/bp`, engine BP5. That is a future conversion action, not performed here.

Required prep dependencies are the two d01 inputs plus the three Thompson tables above and the archived namelist. Existing runner copies the tables (never symlinks them). A/B/C must retain one chosen I/O/namelist/placement setting; no recovery or execution occurred.
