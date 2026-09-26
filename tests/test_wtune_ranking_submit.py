"""Regressions from C3 jobs 1218/1225 (2026-09-26)."""
import re
import subprocess
from pathlib import Path

import pytest

import wtune_tune

SBATCH = Path(__file__).resolve().parents[1] / "campaign" / "strict_campaign.sbatch"


def _config_file_read_loop():
    """The sbatch's own CONFIG_FILE read loop, extracted verbatim (while ... done)."""
    text = SBATCH.read_text()
    m = re.search(r"(  while IFS=\$'\\037' read -r cid .*?\n  done < <\(.*?\)\n)", text, re.S)
    assert m, "CONFIG_FILE read loop not found in strict_campaign.sbatch"
    return m.group(1)


def test_config_file_loop_keeps_empty_fields_in_place(tmp_path):
    # generated configs leave patch_x/patch_y empty; tab-IFS used to collapse them and shift
    # config_source ('anchor') into tile_sz_y.
    tsv = ("config_id\tranks\tomp\tnproc_x\tnproc_y\tnumtiles\tdist\tblocktime\tpatch_x\tpatch_y\t"
           "numtiles_x\tnumtiles_y\ttile_sz_x\ttile_sz_y\ttile_strategy\tconfig_source\n"
           "ANCHOR\t288\t1\t12\t24\t8\tblock\t0\t\t\t0\t0\t0\t0\t0\tanchor\n")
    src = tmp_path / "rows.tsv"
    src.write_text(tsv)
    loop = _config_file_read_loop().replace(
        re.search(r"done < <\(.*?\)", _config_file_read_loop()).group(0),
        "done < <(cat %s | tr '\\t' '\\037')" % src)
    script = ("declare -A CFG_RANKS CFG_OMP CFG_NX CFG_NY CFG_NUMTILES CFG_DIST CFG_BLOCKTIME\n"
              "declare -A CFG_NTILES_X CFG_NTILES_Y CFG_TSZ_X CFG_TSZ_Y CFG_TSTRAT CFG_SRC\n"
              + loop +
              'echo "${CFG_NX[ANCHOR]} ${CFG_TSZ_Y[ANCHOR]} ${CFG_TSTRAT[ANCHOR]} ${CFG_SRC[ANCHOR]}"\n')
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True).stdout
    assert out.split() == ["12", "0", "0", "anchor"]


def _cmd(spinup_minutes):
    return wtune_tune.build_ranking_sbatch_command(
        "conus25", 4, 72, "01:00:00", 1800, 2, 50, "/x/configs.json", "/x/out", spinup_minutes)


def test_spinup_minutes_never_contains_a_comma_inside_export():
    cmd = _cmd("60:122")
    export = re.search(r"--export=(\S+)", cmd).group(1)
    pairs = export.split(",")
    assert "SPINUP_MINUTES=60:122" in pairs
    assert all("=" in p or p == "ALL" for p in pairs)


def test_spinup_minutes_with_comma_is_refused():
    with pytest.raises(ValueError, match="sbatch --export"):
        _cmd("60,122")
