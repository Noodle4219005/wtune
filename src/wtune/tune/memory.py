"""Memory per node of a probe, so that layouts which cannot fit are not submitted.

per node = bytes per grid cell x cells per node + bytes per rank x ranks per node; the two
coefficients are machine.measured values (isc21 on f1-arm: 1600 B and 0.94 GiB). Once a probe of
this case has a MaxRSS, every prediction is scaled by the largest measured / predicted ratio; a
probe killed for memory rules out every layout with no more nodes and no fewer ranks per node."""
from __future__ import annotations

import wtune.machine

GIB = 2.0 ** 30
PRIOR_LIMIT = 1.3       # before any measurement: skip only what is clearly too big
CALIBRATED_LIMIT = 0.95


class MemoryModel:
    def __init__(self, cells: float, node_gib: float, bytes_per_cell: float, gib_per_rank: float):
        self.cells, self.node_gib = cells, node_gib
        self.bytes_per_cell, self.gib_per_rank = bytes_per_cell, gib_per_rank
        self.ratio = None                  # measured / predicted, once a probe has a MaxRSS
        self.oom: list = []                # (nodes, ranks per node) of probes killed for memory

    def _raw_gib(self, nodes: int, ranks_per_node: int) -> float:
        return self.cells / nodes * self.bytes_per_cell / GIB + ranks_per_node * self.gib_per_rank

    def predict_gib(self, nodes: int, ranks_per_node: int) -> float:
        return self._raw_gib(nodes, ranks_per_node) * (self.ratio or 1.0)

    def learn(self, nodes: int, ranks_per_node: int, maxrss_mb: float | None) -> None:
        """A successful probe's MaxRSS; a value far below the model is not WRF's (a launcher
        that hides the ranks from Slurm reports its batch shell) and is ignored."""
        if not maxrss_mb:
            return
        r = maxrss_mb / 1024.0 / self._raw_gib(nodes, ranks_per_node)
        if r >= 0.1:
            self.ratio = max(self.ratio or 0.0, r)

    def learn_oom(self, nodes: int, ranks_per_node: int) -> None:
        self.oom.append((nodes, ranks_per_node))

    def why_not(self, nodes: int, ranks_per_node: int) -> str | None:
        """None when the layout is expected to fit, else the reason it is skipped."""
        for n, r in self.oom:
            if nodes <= n and ranks_per_node >= r:
                return ("memory: N=%d with %d ranks/node was killed for memory"
                        % (n, r))
        limit = CALIBRATED_LIMIT if self.ratio else PRIOR_LIMIT
        need = self.predict_gib(nodes, ranks_per_node)
        if need > limit * self.node_gib:
            return ("memory: predicted %.0f GiB/node > %.0f%% of %.0f GiB (%s)"
                    % (need, limit * 100, self.node_gib,
                       "scaled to this case's MaxRSS" if self.ratio else "prior, no MaxRSS yet"))
        return None


def case_cells(case_json: dict) -> float:
    """3-D grid cells of every active domain; 0 when the case json does not say."""
    max_dom = case_json.get("max_dom", 1)
    we = case_json.get("e_we_list") or [case_json.get("e_we")]
    sn = case_json.get("e_sn_list") or [case_json.get("e_sn")]
    nz = case_json.get("e_vert")
    try:
        return float(sum(w * s for w, s in zip(we[:max_dom], sn[:max_dom])) * nz)
    except TypeError:
        return 0.0


def model_for(case_json: dict, node_gib: float) -> MemoryModel | None:
    """The case's model on nodes of `node_gib`; None when either is unknown."""
    cells = case_cells(case_json)
    if node_gib <= 0 or cells <= 0:
        return None
    return MemoryModel(cells, node_gib, wtune.machine.measured("mem_bytes_per_cell"),
                       wtune.machine.measured("mem_gib_per_rank"))
