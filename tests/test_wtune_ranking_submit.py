"""Regressions from C3 jobs 1218/1225 (2026-09-26)."""
import re
import subprocess
from pathlib import Path

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


def test_adios2_restart_xml_selects_bp4_for_every_domain(tmp_path):
    # WRF's ADIOS2 reader inquires Times without BeginStep; BP5 (default) fails there (job 1273).
    text = SBATCH.read_text()
    fn = re.search(r"(write_adios2_restart_xml\(\) \{.*?\n\})", text, re.S).group(1)
    script = ("RANKING_RESTART_IO_FORM=14\n" + fn +
              "\nwrite_adios2_restart_xml %s 2019-11-26_19:00:00 3\n" % tmp_path)
    subprocess.run(["bash", "-c", script], check=True)
    xml = (tmp_path / "adios2.xml").read_text()
    for d in (1, 2, 3):
        assert '<io name="wrfrst_d%02d_2019-11-26_19:00:00">' % d in xml
    assert xml.count('<engine type="BP4"/>') == 3
    subprocess.run(["bash", "-c", script.replace("=14\n", "=2\n").replace(str(tmp_path), str(tmp_path / "n"))],
                   check=True)
    assert not (tmp_path / "n" / "adios2.xml").exists()
