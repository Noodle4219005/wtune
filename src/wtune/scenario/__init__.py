"""scenario -- pluggable "scenario generators" that produce WRF forcing (as WPS intermediate
files) for fictional dates, behind one interface (base.ScenarioGenerator).

Submodules:
    base              -- ScenarioRequest/Caps/ForcingSet dataclasses, the ScenarioGenerator ABC,
                          and the generator registry (register/get_generator/list_generators).
    wps_intermediate   -- pure-numpy reader/writer for the WPS intermediate file format (v5).
    fill               -- computes fields_missing against a real metgrid run's field list and
                          merges an analog day's records into a primary ForcingSet's files.
    config             -- parse_scenario(dict) -> ScenarioConfig, with the license gate applied.
    cli                -- `wtune scenario ...` subcommands (list / generate / inspect).
    generators/        -- one module per generator (analog, aifs, and not-yet-implemented stubs
                          registered so `wtune scenario list` and the license gate work for them).
"""
