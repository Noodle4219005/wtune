"""wtune.tune.memory: which layouts are expected to fit on a node."""
import wtune.tune.memory as mem

GIB = mem.GIB


def _model(node_gib=200.0):
    # 300 GiB of grid in total, 1 GiB per rank
    return mem.MemoryModel(cells=300 * GIB / 1000.0, node_gib=node_gib, bytes_per_cell=1000.0,
                           gib_per_rank=1.0)


def test_a_maxrss_scales_every_prediction_and_tightens_the_limit():
    m = _model()
    m.learn(4, 72, 1.2 * (75 + 72) * 1024)            # measured 20% above the model
    assert m.ratio == 1.2
    assert "MaxRSS" in m.why_not(3, 72)               # 1.2 x 172 > 0.95 x 200
    assert m.why_not(3, 36) is None                   # fewer ranks per node: 1.2 x 136 fits


def test_a_tiny_maxrss_is_not_the_model_s_and_is_ignored():
    m = _model()
    m.learn(4, 72, 6.0)                               # a batch shell's 6 MB
    assert m.ratio is None


def test_an_oom_rules_out_no_more_nodes_and_no_fewer_ranks():
    m = _model(node_gib=10000.0)
    m.learn_oom(3, 72)
    assert "killed" in m.why_not(2, 72) and "killed" in m.why_not(3, 72)
    assert m.why_not(4, 72) is None and m.why_not(3, 36) is None


def test_case_cells_sums_the_active_domains():
    case = dict(max_dom=2, e_we_list=[100, 50, 9], e_sn_list=[10, 20, 9], e_vert=5)
    assert mem.case_cells(case) == (100 * 10 + 50 * 20) * 5
    assert mem.case_cells(dict(e_we=10, e_sn=10)) == 0.0
    assert mem.model_for(case, 0) is None
