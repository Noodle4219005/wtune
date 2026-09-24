#!/usr/bin/env bash
# G0 dispatch gate, reusable. Takes an exe path.
# Two earlier versions of this gate returned a clean-looking "0 wrapped, 0 unwrapped":
#   1. it disassembled module_em_mp_solve_em_, but solve_em is not in a module (symbol is
#      solve_em_), so the disassembly was empty;
#   2. the pattern was "call " but objdump 2.30 here prints "callq".
# Both times a counter returned the same 0 for "absent" and for "nothing to read". So the
# gate now proves it can read before it reports.
set -uo pipefail
X=${1:?usage: gate0.sh <exe>}
D=$(mktemp /tmp/gate0.XXXXXX)
objdump -d "$X" > "$D" 2>/dev/null
NI=$(wc -l < "$D")
echo "  disassembly lines: $NI"
[ "$NI" -gt 1000000 ] || { echo "  GATE UNUSABLE: disassembly too small"; rm -f "$D"; exit 9; }
echo "  wrapped call sites:"
grep -oE "callq +[0-9a-f]+ <__wrap_module_[a-z_0-9]*>" "$D" | grep -oE "__wrap_module_[a-z_0-9]*" \
  | sort | uniq -c | sed 's/^/    /'
NW=$(grep -cE "callq +[0-9a-f]+ <__wrap_module_[a-z_0-9]*>" "$D")
NR=$(grep -cE "callq +[0-9a-f]+ <module_(microphysics|radiation|pbl|surface|cumulus)_driver_mp_(microphysics|radiation|pre_radiation|pbl|surface|cumulus)_driver_>" "$D")
echo "  total wrapped=$NW   unwrapped direct calls=$NR (must be 0)"
echo "  callers:"
awk '/^[0-9a-f]+ <.*>:/{fn=$2} /callq +[0-9a-f]+ <__wrap_module_/{print fn}' "$D" | sort | uniq -c | sed 's/^/    /'
rm -f "$D"
[ "$NW" -ge 6 ] && [ "$NR" -eq 0 ] && { echo "  G0 PASS"; exit 0; }
echo "  G0 FAIL"; exit 6
