#!/usr/bin/env python3
"""Generate a linker-wrapper argument map from a preprocessed Fortran driver.

The .F source is the wrong input: its SUBROUTINE statement can carry dummies
inside preprocessor blocks that this build does not compile.  Use the compiled
.f90 interface instead.

Usage:
    gen_argmap.py <driver.f90> <subroutine> <output-prefix>

The output prefix is the path without the final ``.h``.  ``wt_argmap`` selects
the historical microphysics profile and ``wt_argmap_rad`` selects the RRTMG
radiation profile (and therefore emits WTR_* names).
"""
import re
import sys


MICRO_W3 = [
    'th', 'pi_phy', 'p', 'w', 'dz8w', 'qv_curr', 'qc_curr', 'qr_curr',
    'qi_curr', 'qs_curr', 'qg_curr', 'qni_curr', 'qnr_curr',
]
MICRO_W2 = [
    'rainnc', 'rainncv', 'snownc', 'snowncv', 'graupelnc', 'graupelncv',
    'sr',
]
MICRO_OPT3 = ['refl_10cm', 're_cloud', 're_ice', 're_snow']

# Strictly column-local RRTMG candidates from radiation_driver.f90.  Optional
# fields are named here so the R1 wrapper can include them only when present;
# the map itself must still resolve every compiled formal's index.
RAD_W3 = [
    'dz8w', 'o3rad', 'p8w', 'p', 'pi', 'rho',
    'rthraten', 'rthratenlw', 'rthratensw', 'rthratenlwc', 'rthratenswc',
    't8w', 't', 'cldfra', 'cldfra_mp_all', 'ccldfra', 'qcconv',
    'qiconv', 'lradius', 'iradius', 'cldfra_dp', 'cldfra_sh', 'cldfra_bl',
    're_cloud', 're_ice', 're_snow', 'f_ice_phy', 'f_rain_phy',
    'qv', 'qc', 'qr', 'qi', 'qi2', 'qi3', 'qs', 'qg', 'qh', 'qndrop',
    'qc_bl', 'qi_bl', 'qnifa', 'qnwfa', 'qnbca', 'aod5503d', 'taod5503d',
    'tauaer300', 'tauaer400', 'tauaer600', 'tauaer999',
    'gaer300', 'gaer400', 'gaer600', 'gaer999',
    'waer300', 'waer400', 'waer600', 'waer999',
    'tauaerlw1', 'tauaerlw2', 'tauaerlw3', 'tauaerlw4',
    'tauaerlw5', 'tauaerlw6', 'tauaerlw7', 'tauaerlw8',
    'tauaerlw9', 'tauaerlw10', 'tauaerlw11', 'tauaerlw12',
    'tauaerlw13', 'tauaerlw14', 'tauaerlw15', 'tauaerlw16',
    'cw_rad',
]
RAD_W2 = [
    'albedo', 'emiss', 'glw', 'gsw', 'rlwtoa', 'snow', 'tsk',
    'xice', 'xland', 'xlat', 'xlong', 'coszen', 'hrang', 'swdown',
    'swdownc', 'swddir', 'swddni', 'swddif', 'swddirc', 'swddnic',
    'lwupt', 'lwuptc', 'lwuptcln', 'lwdnt', 'lwdntc', 'lwdntcln',
    'lwupb', 'lwupbc', 'lwupbcln', 'lwdnb', 'lwdnbc', 'lwdnbcln',
    'swupt', 'swuptc', 'swuptcln', 'swdnt', 'swdntc', 'swdntcln',
    'swupb', 'swupbc', 'swupbcln', 'swdnb', 'swdnbc', 'swdnbcln',
    'lwcf', 'swcf', 'olr',
    'alswvisdir', 'alswvisdif', 'alswnirdir', 'alswnirdif',
    'swvisdir', 'swvisdif', 'swnirdir', 'swnirdif',
    'cldt',
    'acswupt', 'acswuptc', 'acswdnt', 'acswdntc', 'acswupb', 'acswupbc',
    'acswdnb', 'acswdnbc', 'aclwupt', 'aclwuptc', 'aclwdnt', 'aclwdntc',
    'aclwupb', 'aclwupbc', 'aclwdnb', 'aclwdnbc',
    # RRTMG aerosol/FARMS and 2-D diagnostic formals, when present.
    'aodtot', 'aod5502d', 'angexp2d', 'aerssa2d', 'aerasy2d', 'taod5502d',
    'sw_zbbcddir', 'sw_dirdflux', 'sw_difdflux',
]

GEOM = [
    'ids', 'ide', 'jds', 'jde', 'kds', 'kde', 'ims', 'ime', 'jms', 'jme',
    'kms', 'kme', 'i_start', 'i_end', 'j_start', 'j_end', 'kts', 'kte',
    'num_tiles',
]
MICRO_GEOM = [
    'ids', 'ide', 'jds', 'jde', 'kds', 'kde', 'ims', 'ime', 'jms', 'jme',
    'kms', 'kme', 'ips', 'ipe', 'jps', 'jpe', 'kps', 'kpe', 'i_start',
    'i_end', 'j_start', 'j_end', 'kts', 'kte', 'num_tiles',
]
MICRO_SCALARS = [
    'dt', 'dx', 'mp_physics', 'spec_zone', 'itimestep', 'f_qv', 'f_qc',
    'f_qr', 'f_qi', 'f_qs', 'f_qg', 'f_qni', 'f_qnr', 'diagflag',
    'do_radar_ref', 'has_reqc', 'has_reqi', 'has_reqs', 'specified',
    'channel_switch',
]
RAD_SCALARS = [
    'dt', 'radt', 'stepra', 'itimestep', 'ra_call_offset', 'lw_physics',
    'sw_physics', 'icloud', 'o3input', 'aer_opt', 'swint_opt', 'solar_opt',
    'curr_secs', 'adapt_step_flag', 'radtacttime', 'id', 'levsiz', 'n_ozmixm',
    'f_qv', 'f_qc', 'f_qr', 'f_qi', 'f_qs', 'f_qg', 'f_qh', 'f_qndrop',
    'f_qnifa', 'f_qnwfa', 'f_qi2', 'f_qi3', 'icloud_bl', 'is_cammgmp_used',
    'cu_physics', 'shcu_physics', 'mp_physics', 'slope_rad', 'topo_shading',
    'sw_eclipse',
]

# These are intentionally not in the strict whitelist: ozmixm is 4-D and the
# *_flx fields have kme+2 levels.  The R1 wrapper has dedicated pack classes
# for them and needs their compiled positions in the generated map.
RAD_EXTRA = [
    'ozmixm',
    'swupflx', 'swupflxc', 'swdnflx', 'swdnflxc',
    'lwupflx', 'lwupflxc', 'lwdnflx', 'lwdnflxc',
]


def read_interface(src, subroutine):
    lines = open(src).read().split('\n')
    pat = re.compile(r'\s*SUBROUTINE\s+' + re.escape(subroutine) + r'\b', re.I)
    i0 = next(n for n, line in enumerate(lines) if pat.match(line))

    buf = ''
    for line in lines[i0:]:
        s = line.split('!')[0].rstrip()
        if s.strip().startswith('&'):
            s = s.strip()[1:]
        buf += s.rstrip('&')
        if buf.count('(') == buf.count(')') > 0:
            break
    names = [a.strip().lower() for a in
             buf[buf.index('(') + 1:buf.rindex(')')].split(',') if a.strip()]
    assert all(re.fullmatch(r'\w+', n) for n in names), names
    idx = {n: k for k, n in enumerate(names)}

    # Reassemble specification statements and classify only exact dense WRF
    # shapes.  Deliberately do not treat 4-D or kme+N arrays as stageable.
    stmts, cur = [], ''
    for line in lines[i0 + 1:]:
        s = line.split('!')[0].strip()
        if not s:
            continue
        if not cur and re.match(r'(channel\s*=|if\s*\()', s, re.I):
            break
        cur += ' ' + s.rstrip('&')
        if not s.endswith('&'):
            stmts.append(cur.strip())
            cur = ''
    sp = r'\s*'
    d3 = re.compile(r'DIMENSION' + sp + r'\(' + sp + r'ims' + sp + ':' + sp +
                    r'ime' + sp + ',' + sp + r'kms' + sp + ':' + sp + r'kme' +
                    sp + ',' + sp + r'jms' + sp + ':' + sp + r'jme' + sp + r'\)', re.I)
    d2 = re.compile(r'DIMENSION' + sp + r'\(' + sp + r'ims' + sp + ':' + sp +
                    r'ime' + sp + ',' + sp + r'jms' + sp + ':' + sp + r'jme' + sp + r'\)', re.I)
    a3, a2 = set(), set()
    for stmt in stmts:
        if '::' not in stmt:
            continue
        spec, vars_ = stmt.split('::', 1)
        if not re.match(r'\s*REAL', spec, re.I):
            continue
        vs = [re.sub(r'\(.*', '', v).strip().lower()
              for v in re.split(r',(?![^()]*\))', vars_) if v.strip()]
        target = a3 if d3.search(spec) else a2 if d2.search(spec) else None
        if target is not None:
            for v in vs:
                if v in idx:
                    target.add(v)
    return names, idx, a3, a2


def list_literal(values):
    return '{' + ','.join(str(x) for x in values) + '}'


def emit(src, subroutine):
    names, idx, strict3, strict2 = read_interface(src, subroutine)
    is_rad = subroutine.lower() == 'radiation_driver'
    macro = 'WTR' if is_rad else 'WT'
    symbol = macro.lower()
    if is_rad:
        w3, w2, opt3 = RAD_W3, RAD_W2, []
        geom, scalars = GEOM, RAD_SCALARS
        profile_comment = 'RRTMG strict-shape whitelist; optional fields are checked by the wrapper.'
    else:
        w3, w2, opt3 = MICRO_W3, MICRO_W2, MICRO_OPT3
        geom, scalars = MICRO_GEOM, MICRO_SCALARS
        profile_comment = 'Thompson whitelist: what CASE(THOMPSON) hands to mp_gt_driver. Indices are compiled positions.'

    for v in w3 + opt3:
        assert v in strict3, (v, 'not strict 3D')
    for v in w2:
        assert v in strict2, (v, 'not strict 2D')
    for v in geom + scalars:
        assert v in idx, (v, 'not an argument of ' + subroutine)

    idx3d = sorted(idx[v] for v in strict3)
    idx2d = sorted(idx[v] for v in strict2)
    out = []
    out.append('/* GENERATED by gen_argmap.py from %s -- do not edit. */' % src)
    out.append('#define %s_NARG %d' % (macro, len(names)))
    out.append('#define %s_N3D  %d' % (macro, len(idx3d)))
    out.append('#define %s_N2D  %d' % (macro, len(idx2d)))
    out.append('static const int %s_idx3d[%s_N3D] = %s;' %
               (symbol, macro, list_literal(idx3d)))
    out.append('static const int %s_idx2d[%s_N2D] = %s;' %
               (symbol, macro, list_literal(idx2d)))
    for b in geom:
        out.append('#define %s_A_%-9s %d' % (macro, b.upper(), idx[b]))
    for b in scalars:
        out.append('#define %s_A_%-12s %d' % (macro, b.upper(), idx[b]))
    if is_rad:
        for b in w3 + w2 + RAD_EXTRA:
            out.append('#define %s_A_%-12s %d' % (macro, b.upper(), idx[b]))
    out.append('/* %s */' % profile_comment)
    out.append('#define %s_W3D %d' % (macro, len(w3)))
    out.append('#define %s_W2D %d' % (macro, len(w2)))
    out.append('static const int %s_w3d[%s_W3D] = %s; /* %s */' %
               (symbol, macro, list_literal([idx[v] for v in w3]), ' '.join(w3)))
    out.append('static const int %s_w2d[%s_W2D] = %s; /* %s */' %
               (symbol, macro, list_literal([idx[v] for v in w2]), ' '.join(w2)))
    for v in opt3:
        out.append('#define %s_A_%-12s %d' % (macro, v.upper(), idx[v]))
    out.append('static const char *%s_argname[%s_NARG] = {%s};' %
               (symbol, macro, ','.join('"%s"' % n for n in names)))
    return '\n'.join(out) + '\n'


def main():
    if len(sys.argv) != 4:
        raise SystemExit('usage: gen_argmap.py <driver.f90> <subroutine> <output-prefix>')
    src, subroutine, out_prefix = sys.argv[1:]
    out_path = out_prefix if out_prefix.endswith('.h') else out_prefix + '.h'
    with open(out_path, 'w') as f:
        f.write(emit(src, subroutine))


if __name__ == '__main__':
    main()
