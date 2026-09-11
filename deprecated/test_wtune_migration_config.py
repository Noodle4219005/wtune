import unittest

from wtune_migration_config import (DEFAULTS, MigrationConfigError,
                                    from_environment, manifest_metadata,
                                    validate_case)


class MigrationConfigTests(unittest.TestCase):
    def test_defaults(self):
        config = from_environment({})
        self.assertEqual(config, DEFAULTS)
        self.assertEqual(manifest_metadata(config), {
            'activation': 'disabled', 'actionable': False,
            'provider': 'pool', 'ownership': 'rect', 'scope': 'single-domain',
            'fail_closed': True,
        })

    def test_strict_values_and_selection(self):
        env = {'WTUNE_RUNTIME_MIGRATION': '1',
               'WTUNE_MIGRATION_PROVIDER': 'dynamic',
               'WTUNE_MIGRATION_OWNERSHIP': 'block'}
        config = from_environment(env)
        self.assertEqual(config.provider, 'dynamic')
        self.assertEqual(config.ownership, 'block')
        for key, value in [('WTUNE_RUNTIME_MIGRATION', 'true'),
                           ('WTUNE_MIGRATION_FAIL_CLOSED', '2'),
                           ('WTUNE_MIGRATION_PROVIDER', 'bad'),
                           ('WTUNE_MIGRATION_OWNERSHIP', 'bad'),
                           ('WTUNE_MIGRATION_SCOPE', 'all')]:
            with self.assertRaises(MigrationConfigError):
                from_environment(dict(env, **{key: value}))

    def test_enabled_requires_fail_closed_and_nested_case_stays_rejected(self):
        with self.assertRaises(MigrationConfigError):
            from_environment({'WTUNE_RUNTIME_MIGRATION': '1',
                              'WTUNE_MIGRATION_FAIL_CLOSED': '0'})
        config = from_environment({})
        with self.assertRaises(MigrationConfigError):
            validate_case(config, {'max_dom': 2})

    def test_enabled_is_blocked_and_never_actionable(self):
        config = from_environment({'WTUNE_RUNTIME_MIGRATION': '1'})
        metadata = manifest_metadata(config)
        self.assertEqual(metadata['activation'], 'blocked-no-runtime-hook')
        self.assertFalse(metadata['actionable'])


if __name__ == '__main__':
    unittest.main()
