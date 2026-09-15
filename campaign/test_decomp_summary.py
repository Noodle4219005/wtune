import csv
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import decomp_summary as dsum  # noqa: E402

HEADER = ("run_id,case_base,compliance_class,e_we,e_sn,n_ranks,omp_threads,nproc_x,nproc_y,"
          "numtiles,distribution,kmp_blocktime,s_per_step_median,notes")


def _rows(text):
    return list(csv.DictReader(io.StringIO(HEADER + "\n" + text)))


def test_template_proposal_and_confirmation_pool_by_decomposition():
    rows = _rows(
        "x_C02_r1_1,x,STRICT,1000,1000,448,1,8,56,8,cyclic,0,1.30,config=C02\n"
        "x_R01D0003_r1_1,x,STRICT,1000,1000,224,2,8,28,16,block,0,1.31,"
        "config=R01D0003 numtiles_x=0 numtiles_y=0 tile_sz_x=0 tile_sz_y=0 tile_strategy=0\n"
        "x_CFR01D0003_r1_1,x,STRICT,1000,1000,224,2,8,28,16,block,0,1.29,config=CFR01D0003\n"
        "x_R01D0004_r1_1,x,STRICT,1000,1000,224,2,8,28,16,block,0,,config=R01D0004\n"
        "y_C02_r1_1,y,STRICT,1000,1000,448,1,8,56,8,cyclic,0,0.10,config=C02\n")
    ranked = dsum.summarize(rows, "x")
    assert [d["ids"] for d in ranked] == [["C02"], ["R01D0003"]]
    assert ranked[1]["n"] == 2 and abs(ranked[1]["s_median"] - 1.30) < 1e-9


def test_tile_geometry_keeps_decompositions_apart():
    rows = _rows(
        "x_A_r1_1,x,STRICT,901,901,224,2,8,28,12,block,0,0.26,config=A\n"
        "x_B_r1_1,x,STRICT,901,901,224,2,8,28,12,block,0,0.30,"
        "config=B numtiles_x=3 numtiles_y=4 tile_strategy=0\n")
    assert len(dsum.summarize(rows, "x")) == 2


def test_confirm_configs_have_nonempty_patch_and_cf_ids(tmp_path):
    rows = _rows("x_R01D0002_r1_1,x,STRICT,901,901,224,2,8,28,12,block,default,0.26,config=R01D0002\n")
    ledger = tmp_path / "l.csv"
    ledger.write_text(HEADER + "\n" + "\n".join(",".join(r[k] for k in HEADER.split(",")) for r in rows) + "\n")
    out = tmp_path / "cf.json"
    assert dsum.main(["--case", "x", "--ledger", str(ledger), "--confirm-out", str(out)]) == 0
    cfg = json.loads(out.read_text())["configs"][0]
    assert cfg["id"] == "CFR01D0002"
    assert (cfg["patch_x"], cfg["patch_y"]) == (112, 32)
    assert cfg["blocktime"] == "default" and cfg["tile_shape"] == "1d_y"
