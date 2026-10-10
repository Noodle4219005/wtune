#!/usr/bin/env python3
"""wtune_iofields.py -- WRF iofields files (`iofields_filename`): parse, count, merge lines.

WRF re-parses the file on rank 0 at every input/output call (frame/module_domain.F
modify_io_masks1) with 5 x lines + fields get_fieldstr calls, while every other rank waits.
Merging adjacent lines of one `op:kind:stream` prefix gives the same masks with fewer calls
(climate-ea50, scc jobs 3142-3145: bit-identical, 10% faster). Stdlib only, Python 3.6."""
import collections
import os
from typing import List, Sequence, Tuple

MAX_LINE_CHARS = 7000       # WRF reads a line into CHARACTER*8000 (module_domain.F:1054)
MAX_LINE_FIELDS = 900       # get_fieldstr's MAX_DEXES = 1000 (module_domain.F:998)

# raw: the line as written; op None: a comment, blank or unrecognised line, kept as it is
Entry = collections.namedtuple("Entry", "raw op kind stream fields")


def _entry(line: str) -> Entry:
    parts = line.rstrip().split(":", 3)
    fields = parts[3].split(",") if len(parts) == 4 else []
    # WRF stops a line's field list at an empty name: such a line is left alone
    if parts[0].strip() not in ("+", "-") or not fields or not all(f.strip() for f in fields):
        return Entry(line, None, None, None, ())
    return Entry(line, parts[0], parts[1], parts[2], tuple(fields))


def parse(text: str) -> List[Entry]:
    """[Entry, ...], one per line of an iofields file."""
    return [_entry(line) for line in text.splitlines()]


def counts(text: str) -> Tuple[int, int]:
    """(lines, fields) WRF parses: the `op:kind:stream:fields` lines and their field names."""
    entries = [e for e in parse(text) if e.op]
    return len(entries), sum(len(e.fields) for e in entries)


def parse_calls(text: str) -> int:
    """get_fieldstr calls of one parse of the file: 5 per line + 1 per field."""
    lines, fields = counts(text)
    return 5 * lines + fields


def _key(e: Entry) -> Tuple[str, str, str]:
    return (e.op.strip(), e.kind.strip().lower(), e.stream.strip())


def _groups(entries: Sequence[Entry]) -> List[List[Entry]]:
    """Runs of adjacent entries that share a prefix; an unrecognised line is its own run."""
    runs = []
    for e in entries:
        if e.op and runs and runs[-1][0].op and _key(runs[-1][0]) == _key(e):
            runs[-1].append(e)
        else:
            runs.append([e])
    return runs


def _packed(run: Sequence[Entry]) -> List[str]:
    """The run's fields on as few lines as WRF's limits allow, under the first line's prefix."""
    prefix = "%s:%s:%s:" % run[0][1:4]
    lines, cur, size = [], [], len(prefix)
    for f in (f for e in run for f in e.fields):
        if cur and (len(cur) >= MAX_LINE_FIELDS or size + len(f) + 1 > MAX_LINE_CHARS):
            lines.append(prefix + ",".join(cur))
            cur, size = [], len(prefix)
        cur.append(f)
        size += len(f) + 1
    return lines + [prefix + ",".join(cur)]


def merged(text: str) -> str:
    """`text` with adjacent lines of the same `op:kind:stream` merged; field order and
    spelling, and every other line, are kept. The input itself when nothing merges."""
    out, changed = [], False
    for run in _groups(parse(text)):
        lines = _packed(run) if len(run) > 1 else None
        if lines and len(lines) < len(run):
            out += lines
            changed = True
        else:
            out += [e.raw for e in run]
    if not changed:
        return text
    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


def merge_in_dir(run_dir: str, names: Sequence[str]) -> List[Tuple[str, int, int]]:
    """Replace each iofields file `names` of run_dir (a link into the case, usually) by a
    regular file holding its merged text, when merging changes it -> [(name, lines before,
    lines after)]. Never writes through a link."""
    done = []
    for name in names:
        path = os.path.join(run_dir, name)
        outside = os.path.isabs(name) or os.path.normpath(name).split(os.sep)[0] == ".."
        if outside or not os.path.isfile(path):
            continue
        with open(path) as f:
            text = f.read()
        new = merged(text)
        if new is text:
            continue
        os.remove(path)
        with open(path, "w") as f:
            f.write(new)
        done.append((name, counts(text)[0], counts(new)[0]))
    return done


def merge_log_lines(done: Sequence[Tuple[str, int, int]]) -> List[str]:
    return ["  iofields: %s %d -> %d line(s) (same fields; WRF parses it at every I/O call)" % d
            for d in done]
