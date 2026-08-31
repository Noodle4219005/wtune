"""Configuration for the deliberately non-actionable runtime migration advisor."""

import os
from collections import namedtuple


MigrationConfig = namedtuple('MigrationConfig',
                             'runtime_migration provider ownership scope fail_closed')
DEFAULTS = MigrationConfig(False, 'pool', 'rect', 'single-domain', True)
MigrationConfigError = ValueError

_NAMES = {
    'runtime_migration': 'WTUNE_RUNTIME_MIGRATION',
    'provider': 'WTUNE_MIGRATION_PROVIDER',
    'ownership': 'WTUNE_MIGRATION_OWNERSHIP',
    'scope': 'WTUNE_MIGRATION_SCOPE',
    'fail_closed': 'WTUNE_MIGRATION_FAIL_CLOSED',
}


def _value(environ, name, default):
    raw = environ.get(name, default)
    if raw not in ('0', '1'):
        raise MigrationConfigError('{} must be 0 or 1'.format(name))
    return raw == '1'


def from_environment(environ=None):
    environ = os.environ if environ is None else environ
    config = MigrationConfig(
        _value(environ, _NAMES['runtime_migration'], '0'),
        environ.get(_NAMES['provider'], 'pool'),
        environ.get(_NAMES['ownership'], 'rect'),
        environ.get(_NAMES['scope'], 'single-domain'),
        _value(environ, _NAMES['fail_closed'], '1'))
    if config.provider not in ('pool', 'dynamic'):
        raise MigrationConfigError('WTUNE_MIGRATION_PROVIDER must be pool or dynamic')
    if config.ownership not in ('rect', 'block'):
        raise MigrationConfigError('WTUNE_MIGRATION_OWNERSHIP must be rect or block')
    if config.scope != 'single-domain':
        raise MigrationConfigError('WTUNE_MIGRATION_SCOPE must be single-domain')
    if config.runtime_migration and not config.fail_closed:
        raise MigrationConfigError('enabled runtime migration requires fail-closed=1')
    return config


def validate_case(config, case):
    if int(case.get('max_dom', 1)) != 1:
        raise MigrationConfigError('max_dom=1 is required')
    return case


def manifest_metadata(config):
    return {
        'activation': ('blocked-no-runtime-hook' if config.runtime_migration else 'disabled'),
        'actionable': False,
        'provider': config.provider,
        'ownership': config.ownership,
        'scope': config.scope,
        'fail_closed': config.fail_closed,
    }
