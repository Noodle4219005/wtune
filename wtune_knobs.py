"""WRF namelist knob-advisor rules engine. Stdlib only."""

_CATEGORY_ORDER = (
    'correctness-setup',
    'resolution(dx)',
    'stability',
    'run-length',
    'terrain-latitude',
)

_NUM_SOIL_LSM = {0: 5, 1: 5, 2: 4, 5: 4, 3: 6, 7: 2, 8: 3}


def dx_km(case):
    dx = case.get('dx')
    if dx is None:
        return None
    return float(dx) / 1000.0


def _finding(param, category, tag, enforced, current, recommended, verdict, reason):
    return {
        'param': param,
        'group': category,
        'category': category,
        'tag': tag,
        'enforced': enforced,
        'current': current,
        'recommended': recommended,
        'verdict': verdict,
        'reason': reason,
    }


def _rule_num_soil(case):
    lsm = case.get('sf_surface_physics')
    if lsm not in _NUM_SOIL_LSM:
        return None
    rec = _NUM_SOIL_LSM[lsm]
    cur = case.get('num_soil_layers')
    if cur is not None:
        if cur == rec:
            return _finding('num_soil_layers', 'correctness-setup', 'CORRECTNESS', True,
                            cur, rec, 'OK',
                            'num_soil_layers matches LSM {0}.'.format(lsm))
        return _finding('num_soil_layers', 'correctness-setup', 'CORRECTNESS', True,
                        cur, rec, 'MISMATCH',
                        'WRF will silently override num_soil_layers={0} to {1} for LSM {2} '
                        '(mundo:3588-3612).'.format(cur, rec, lsm))
    return _finding('num_soil_layers', 'correctness-setup', 'CORRECTNESS', True,
                    cur, rec, 'SUGGEST',
                    'Set num_soil_layers={0} for LSM {1}; WRF silently overrides it '
                    'otherwise (mundo:3588-3612).'.format(rec, lsm))


def _rule_pbl_sfclay(case):
    pbl = case.get('bl_pbl_physics')
    sfc = case.get('sf_sfclay_physics')

    if pbl == 10:
        rec, label, enforced, fatal = 10, 'TEMF', True, True
    elif pbl in (5, 6):
        rec, label, enforced, fatal = 5, 'MYNN', False, False
    elif pbl == 2:
        rec, label, enforced, fatal = 2, 'MYJ', True, False
    elif pbl == 1:
        rec, label, enforced, fatal = 1, 'YSU', True, False
    else:
        return None

    if sfc is None:
        reason = 'Set sf_sfclay_physics={0} for {1} PBL'.format(rec, label)
        if fatal:
            reason += ' (WRF fatal if mismatched)'
        if not enforced:
            reason += ' (guidance, not code-enforced)'
        return _finding('sf_sfclay_physics', 'correctness-setup', 'CORRECTNESS',
                        enforced, sfc, rec, 'SUGGEST', reason)

    if sfc == rec:
        return _finding('sf_sfclay_physics', 'correctness-setup', 'CORRECTNESS',
                        enforced, sfc, rec, 'OK',
                        '{0} PBL / sfclay pairing is consistent.'.format(label))

    if label == 'MYNN':
        return _finding('sf_sfclay_physics', 'correctness-setup', 'CORRECTNESS',
                        enforced, sfc, rec, 'SUGGEST',
                        'MYNN PBL is typically paired with sf_sfclay_physics=5 '
                        '(guidance, not code-enforced).')

    if label == 'TEMF':
        return _finding('sf_sfclay_physics', 'correctness-setup', 'CORRECTNESS',
                        enforced, sfc, rec, 'MISMATCH',
                        'TEMF requires sf_sfclay_physics=10; mismatch is fatal (WRF fatal).')

    return _finding('sf_sfclay_physics', 'correctness-setup', 'CORRECTNESS',
                    enforced, sfc, rec, 'MISMATCH',
                    '{0} suite default pairing is {1}; mismatch usually wrong.'.format(label, rec))


def _rule_kmopt5(case):
    if case.get('km_opt') != 5:
        return None
    pbl = case.get('bl_pbl_physics')
    diff_opt = case.get('diff_opt')
    sfclay = case.get('sf_sfclay_physics')
    if pbl is None or diff_opt is None or sfclay is None:
        return None

    violations = []
    if pbl != 0:
        violations.append('bl_pbl_physics={0} (needs 0)'.format(pbl))
    if diff_opt != 2:
        violations.append('diff_opt={0} (needs 2)'.format(diff_opt))
    if sfclay not in (0, 1, 5, 91):
        violations.append('sf_sfclay_physics={0} (needs 0/1/5/91)'.format(sfclay))

    rec = 'bl_pbl_physics=0, diff_opt=2, sf_sfclay_physics in {0,1,5,91}'
    if violations:
        return _finding('km_opt', 'correctness-setup', 'CORRECTNESS', True,
                        5, rec, 'MISMATCH',
                        'SMS-3DTKE requires ' + rec + ' (mundo:356-399); violation: ' +
                        '; '.join(violations))
    return _finding('km_opt', 'correctness-setup', 'CORRECTNESS', True,
                    5, rec, 'OK',
                    'SMS-3DTKE prerequisite options are satisfied.')


def _rule_diff_km_set(case):
    findings = []
    diff_opt = case.get('diff_opt')
    if diff_opt is None or diff_opt == -1:
        findings.append(_finding('diff_opt', 'correctness-setup', 'CORRECTNESS', True,
                                 diff_opt, '1 or 2', 'SUGGEST',
                                 'diff_opt must be set (1 or 2); Registry default -1 means unset.'))
    km_opt = case.get('km_opt')
    if km_opt is None or km_opt == -1:
        findings.append(_finding('km_opt', 'correctness-setup', 'CORRECTNESS', True,
                                 km_opt, '2/3 mesoscale, 5 LES', 'SUGGEST',
                                 'km_opt must be set (2/3 mesoscale, 5 LES); Registry default -1 means unset.'))
    return findings or None


def _rule_o3input(case):
    o3 = case.get('o3input')
    if o3 is None or o3 == 0:
        return None
    lw = case.get('ra_lw_physics')
    sw = case.get('ra_sw_physics')
    if lw is None or sw is None:
        return None
    if lw != 4 and sw != 4:
        return _finding('o3input', 'correctness-setup', 'CORRECTNESS', True,
                        o3, 0, 'SUGGEST',
                        'WRF auto-zeroes o3input for non-RRTMG (mundo:2943-2962); set o3input=0 '
                        'or use RRTMG.')
    return None


def _rule_cu(case):
    dk = dx_km(case)
    if dk is None:
        return None
    cu = case.get('cu_physics')
    shcu = case.get('shcu_physics')

    if dk <= 4.0:
        if cu == 0:
            return _finding('cu_physics', 'resolution(dx)', 'RESULTS', False,
                            cu, 0, 'OK',
                            'At dx={0:.1f} km convection is resolved; cumulus usually 0 '
                            '(best-practice, NOT code-enforced).'.format(dk))
        return _finding('cu_physics', 'resolution(dx)', 'RESULTS', False,
                        cu, 0, 'SUGGEST',
                        'At dx={0:.1f} km convection is resolved; cumulus usually 0 '
                        '(best-practice, NOT code-enforced).'.format(dk))

    if dk >= 10.0:
        rec = 'on(scale-aware GF=3/Tiedtke=6/MSKF=11 or KF=1)'
        if cu in (None, 0):
            return _finding('cu_physics', 'resolution(dx)', 'RESULTS', False,
                            cu, rec, 'SUGGEST',
                            'At dx={0:.1f} km convection is parameterized; recommend {1} '
                            '(best-practice, NOT code-enforced).'.format(dk, rec))
        return _finding('cu_physics', 'resolution(dx)', 'RESULTS', False,
                        cu, rec, 'OK',
                        'Deep convection parameterization is active (best-practice, NOT code-enforced).')

    rec = 'scale-aware (GF=3/Tiedtke=6/MSKF=11) or 0 with shcu_physics>0'
    if cu in (3, 6, 11) or (cu == 0 and shcu is not None and shcu > 0):
        return _finding('cu_physics', 'resolution(dx)', 'RESULTS', False,
                        cu, rec, 'OK',
                        'Scale-aware cumulus or shallow-cu-only at gray-zone dx={0:.1f} km '
                        '(best-practice, NOT code-enforced).'.format(dk))
    return _finding('cu_physics', 'resolution(dx)', 'RESULTS', False,
                    cu, rec, 'SUGGEST',
                    'At gray-zone dx={0:.1f} km recommend {1} '
                    '(best-practice, NOT code-enforced).'.format(dk, rec))


def _rule_radt(case):
    dk = dx_km(case)
    radt = case.get('radt')
    if dk is None or radt is None:
        return None
    rule = min(30, max(1, round(dk)))
    rec = '{0}-{1} min'.format(rule, 2 * rule)

    if rule <= radt <= 2 * rule:
        return _finding('radt', 'resolution(dx)', 'RESULTS', False,
                        radt, rec, 'OK',
                        'radt cadence is consistent with resolution; not code-enforced; results-affecting.')
    if radt > 2 * rule:
        return _finding('radt', 'resolution(dx)', 'RESULTS', False,
                        radt, rec, 'SUGGEST',
                        'Coarse radt cadence; recommended {0}; not code-enforced; results-affecting.'.format(rec))
    return _finding('radt', 'resolution(dx)', 'RESULTS', False,
                    radt, rec, 'SUGGEST',
                    'Finer radt than needed; recommended {0}; not code-enforced; results-affecting.'.format(rec))


def _rule_diff6(case):
    dk = dx_km(case)
    if dk is None:
        return None
    rec = 2 if dk <= 4.0 else 0
    cur = case.get('diff_6th_opt')
    if cur == rec:
        return _finding('diff_6th_opt', 'resolution(dx)', 'RESULTS', False,
                        cur, rec, 'OK',
                        '6th-order diffusion setting is appropriate for dx={0:.1f} km '
                        '(best-practice, NOT code-enforced).'.format(dk))
    return _finding('diff_6th_opt', 'resolution(dx)', 'RESULTS', False,
                    cur, rec, 'SUGGEST',
                    '6th-order diffusion controls small-scale noise at convective scale; '
                    'recommended {0} for dx={1:.1f} km '
                    '(best-practice, NOT code-enforced).'.format(rec, dk))


def _rule_nonhydro(case):
    dk = dx_km(case)
    if dk is None or dk >= 10.0:
        return None
    cur = case.get('non_hydrostatic')
    if cur is True:
        return _finding('non_hydrostatic', 'resolution(dx)', 'RESULTS', False,
                        cur, True, 'OK',
                        'non_hydrostatic=True is appropriate below ~10 km '
                        '(best-practice, NOT code-enforced).')
    if cur is False:
        return _finding('non_hydrostatic', 'resolution(dx)', 'RESULTS', False,
                        cur, True, 'SUGGEST',
                        'Hydrostatic assumption is invalid below ~10 km; non_hydrostatic should be true '
                        '(best-practice, NOT code-enforced).')
    return _finding('non_hydrostatic', 'resolution(dx)', 'RESULTS', False,
                    cur, True, 'SUGGEST',
                    'Set non_hydrostatic=True below ~10 km '
                    '(best-practice, NOT code-enforced).')


def _rule_mp(case):
    dk = dx_km(case)
    if dk is None or dk > 4.0:
        return None
    mp = case.get('mp_physics')
    if mp not in (6, 8):
        return None
    return _finding('mp_physics', 'resolution(dx)', 'DIAGNOSTIC', False,
                    mp, 'double-moment/aerosol-aware (Thompson-aero 28, Morrison 10)',
                    'SUGGEST',
                    'At dx={0:.1f} km double-moment/aerosol-aware microphysics resolve convective '
                    'microphysics; single-moment (WSM6 6, Thompson 8) fine for dx>4 km '
                    '(best-practice, NOT code-enforced).'.format(dk))


def _rule_w_damping(case):
    cur = case.get('w_damping')
    ptop = case.get('p_top_requested')
    high = ptop is not None and ptop < 5000
    if cur == 1:
        reason = 'w_damping=1; recommended for real-data runs.'
        if high:
            reason += ' Especially important for high-top runs (p_top<50 hPa).'
        return _finding('w_damping', 'stability', 'RESULTS', False,
                        cur, 1, 'OK', reason)
    reason = 'w_damping=1 is recommended for real-data runs (best-practice, NOT code-enforced).'
    if high:
        reason = ('Strongly recommend w_damping=1 for high-top real-data runs (p_top<50 hPa); '
                  'dampens gravity-wave reflection (best-practice, NOT code-enforced).')
    return _finding('w_damping', 'stability', 'RESULTS', False,
                    cur, 1, 'SUGGEST', reason)


def _rule_epssm(case):
    cur = case.get('epssm')
    if cur is None:
        return None
    if 0.08 <= cur <= 0.2:
        return _finding('epssm', 'stability', 'RESULTS', False,
                        cur, 0.1, 'OK',
                        'epssm is in the typical real-data range.')
    return _finding('epssm', 'stability', 'RESULTS', False,
                    cur, 0.1, 'SUGGEST',
                    'Typical real-data epssm is 0.1 (best-practice, NOT code-enforced).')


def _rule_gwd(case):
    dk = dx_km(case)
    if dk is None:
        return None
    cur = case.get('gwd_opt')

    if dk >= 10.0:
        if cur in (1, 3):
            return _finding('gwd_opt', 'stability', 'RESULTS', False,
                            cur, '1/3 over significant terrain', 'OK',
                            'Gravity wave drag is active; consider terrain significance '
                            '(best-practice, NOT code-enforced).')
        return _finding('gwd_opt', 'stability', 'RESULTS', False,
                        cur, '1/3 over significant terrain', 'SUGGEST',
                        'Consider gwd_opt=1/3 over significant terrain '
                        '(best-practice, NOT code-enforced).')

    if cur == 0:
        return _finding('gwd_opt', 'stability', 'RESULTS', False,
                        cur, 0, 'OK',
                        'gwd_opt=0 is usually appropriate below ~10 km '
                        '(best-practice, NOT code-enforced).')
    return _finding('gwd_opt', 'stability', 'RESULTS', False,
                    cur, 0, 'SUGGEST',
                    'For dx<10 km, gwd_opt=0 is usually appropriate '
                    '(best-practice, NOT code-enforced).')


def _rule_sst_update(case):
    run = case.get('run_total_hours')
    if run is None:
        return None
    cur = case.get('sst_update')

    if run > 24:
        if cur == 1:
            return _finding('sst_update', 'run-length', 'RESULTS', False,
                            cur, 1, 'OK',
                            'SST update enabled for long run.')
        return _finding('sst_update', 'run-length', 'RESULTS', False,
                        cur, 1, 'SUGGEST',
                        'Long run: update SST from wrflowinp if ocean present & forcing available '
                        '(best-practice, NOT code-enforced).')

    if cur == 1:
        return _finding('sst_update', 'run-length', 'RESULTS', False,
                        cur, 0, 'SUGGEST',
                        'For runs <=24 h, sst_update=0 is typical; not code-enforced.')
    return None


def _rule_tmn_update(case):
    run = case.get('run_total_hours')
    if run is None or run <= 24:
        return None
    lsm = case.get('sf_surface_physics')
    if lsm not in (2, 5):
        return None
    cur = case.get('tmn_update')
    if cur == 1:
        return _finding('tmn_update', 'run-length', 'RESULTS', False,
                        cur, 1, 'OK',
                        'Deep soil temperature update enabled for long Noah/Noah-MP run.')
    return _finding('tmn_update', 'run-length', 'RESULTS', False,
                    cur, 1, 'SUGGEST',
                    'Update deep soil temperature on long Noah/Noah-MP runs '
                    '(best-practice, NOT code-enforced).')


def _rule_bucket_mm(case):
    run = case.get('run_total_hours')
    if run is None or run <= 48:
        return None
    cur = case.get('bucket_mm')
    if cur in (None, -1):
        return _finding('bucket_mm', 'run-length', 'RESULTS', False,
                        cur, 100.0, 'SUGGEST',
                        '~100 mm bounds accumulation-array growth on long runs '
                        '(best-practice, NOT code-enforced).')
    return None


def _rule_prec_acc_dt(case):
    run = case.get('run_total_hours')
    if run is None or run <= 24:
        return None
    cur = case.get('prec_acc_dt')
    if cur in (None, 0):
        return _finding('prec_acc_dt', 'run-length', 'RESULTS', False,
                        cur, 'output interval', 'SUGGEST',
                        'Set to output interval to get per-interval precip '
                        '(best-practice, NOT code-enforced).')
    return None


def _rule_fractional_seaice(case):
    lat = case.get('latitude')
    if lat is None or abs(lat) <= 55:
        return None
    cur = case.get('fractional_seaice')
    if cur == 1:
        return _finding('fractional_seaice', 'terrain-latitude', 'RESULTS', False,
                        cur, 1, 'OK',
                        'Fractional sea ice enabled for high-latitude ocean.')
    return _finding('fractional_seaice', 'terrain-latitude', 'RESULTS', False,
                    cur, 1, 'SUGGEST',
                    'High-latitude ocean: use fractional_seaice=1 '
                    '(best-practice, NOT code-enforced).')


def _rule_terrain_radiation(case):
    return _finding('slope_rad/topo_shading', 'terrain-latitude', 'DIAGNOSTIC', False,
                    None, 'see note', 'SUGGEST',
                    'If domain has complex terrain and dx<=few km, consider slope_rad=1, '
                    'topo_shading=1 (needs terrain-std info not in namelist).')


def _rule_urban(case):
    if case.get('sf_urban_physics') == 0:
        return _finding('sf_urban_physics', 'terrain-latitude', 'DIAGNOSTIC', False,
                        0, '1 (UCM) or 2 (BEP) if urban-focused; else 0', 'SUGGEST',
                        '0 = bulk urban; enable UCM(1)/BEP(2) only for urban-focused runs.')
    return None


_RULES = (
    _rule_num_soil,
    _rule_pbl_sfclay,
    _rule_kmopt5,
    _rule_diff_km_set,
    _rule_o3input,
    _rule_cu,
    _rule_radt,
    _rule_diff6,
    _rule_nonhydro,
    _rule_mp,
    _rule_w_damping,
    _rule_epssm,
    _rule_gwd,
    _rule_sst_update,
    _rule_tmn_update,
    _rule_bucket_mm,
    _rule_prec_acc_dt,
    _rule_fractional_seaice,
    _rule_terrain_radiation,
    _rule_urban,
)


def advise(case):
    findings = []
    for rule in _RULES:
        result = rule(case)
        if result is None:
            continue
        if isinstance(result, list):
            findings.extend(result)
        else:
            findings.append(result)
    return findings


def advise_grouped(case):
    grouped = {category: [] for category in _CATEGORY_ORDER}
    for finding in advise(case):
        grouped.setdefault(finding['category'], []).append(finding)
    return grouped


if __name__ == '__main__':
    pass
