"""Tiny archives only: exact extraction, altered source and path safety."""
import io
import json
import hashlib
from pathlib import Path
import re
import tarfile
import tempfile
import unittest
from plov import prepare_isc21_node_g0 as prep

ARTIFACTS = Path('/home/u5727520/wrf-stack/wtune/artifacts/20260913-node-local-physics/kf/g0-harness-tests')
NAMELIST = '''&time_control
run_minutes=30,
history_interval_s=0,
iofields_filename="unused",
io_form_auxhist24=2,
/
outside_ignored=1,
&domains
time_step=15,
e_we=793,805,1001,
e_sn=853,805,1001,
e_vert=51,51,51,
eta_levels=1.0, 0.9,
0.5, 0.0,
max_dom=3,
/
&physics
cu_physics=1,0,0,
bl_pbl_physics=5,5,5,
sf_sfclay_physics=5,5,5,
mp_physics=8,8,8,
/
'''


class PrepareTests(unittest.TestCase):
    def setUp(self):
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix='prepare-', dir=ARTIFACTS))
        self.archive, self.output = self.root / 'tiny.tar', self.root / 'inputs'
        self.data = {name: NAMELIST.encode() if name == 'namelist.input' else name.encode()
                     for name in prep.MEMBERS}
        self.expected = {name: len(value) for name, value in self.data.items()}

    def make_archive(self, altered=None, extra=None):
        with tarfile.open(self.archive, 'w') as archive:
            for name, value in self.data.items():
                member = tarfile.TarInfo(prep.PREFIX + name)
                member.size = len(value)
                if name == altered:
                    member.type, member.linkname, member.size = tarfile.SYMTYPE, '/etc/passwd', 0
                archive.addfile(member, io.BytesIO(value) if name != altered else None)
            if extra:
                member = tarfile.TarInfo(extra)
                archive.addfile(member, io.BytesIO())

    def test_plan_then_exact_extract_no_retry(self):
        self.make_archive()
        plan = prep.make_plan(self.archive, self.output, self.expected)
        self.assertFalse(self.output.exists())
        result = prep.extract(plan, self.expected)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['archive_sha256'], prep.sha256(self.archive))
        self.assertEqual(result['members'].keys(), self.data.keys())
        for name, value in self.data.items():
            self.assertEqual((self.output / name).read_bytes(), value)
            self.assertEqual((self.output / name).stat().st_mode & 0o222, 0)
        self.assertEqual(json.loads((self.output / 'recovery.json').read_text()), result)
        with self.assertRaisesRegex(ValueError, 'already exists'):
            prep.extract(plan, self.expected)

    def test_namelist_fixed_geometry_io_and_preserved_physics(self):
        text = prep.namelist_g0(NAMELIST)
        for value in ('run_minutes = 5', 'max_dom = 1', 'nproc_x = 7', 'nproc_y = 8',
                      'numtiles = 12', 'time_step = 15', 'io_form_input = 2',
                      'io_form_boundary = 2', 'io_form_history = 2', 'io_form_auxhist24 = 0',
                      'mp_physics = 8,8,8', 'end_minute = 5', 'use_adaptive_time_step = .false.'):
            self.assertIn(value, text)
        self.assertNotIn('outside_ignored', text)
        self.assertNotIn('iofields_filename', text)
        self.assertNotIn('history_interval_s', text)
        with self.assertRaisesRegex(ValueError, 'physical baseline'):
            prep.namelist_g0(NAMELIST.replace('cu_physics=1', 'cu_physics=2'))

    def test_actual_archived_groups_match_frozen_registry(self):
        # Read only the original 4,179-byte member; no scientific input extraction.
        with tarfile.open(prep.ARCHIVE, 'r:') as archive:
            member = archive.getmember(prep.PREFIX + 'namelist.input')
            self.assertEqual(member.size, prep.MEMBERS['namelist.input'])
            original = archive.extractfile(member).read().decode('ascii')
        generated = prep.namelist_g0(original)

        def line_groups(source):
            # Independent state machine: do not repeat the production group regex.
            groups, current = {}, None
            for line in source.splitlines():
                clean = line.split('!', 1)[0].strip()
                if clean.startswith('&'):
                    self.assertIsNone(current, 'nested or swallowed group')
                    current = clean[1:].lower()
                    self.assertNotIn(current, groups)
                    groups[current] = {}
                elif clean == '/':
                    self.assertIsNotNone(current)
                    current = None
                elif current is not None and '=' in clean:
                    key, value = clean.split('=', 1)
                    groups[current][key.strip().lower()] = value.rstrip(',').strip()
            self.assertIsNone(current)
            return groups

        before, after = line_groups(original), line_groups(generated)
        self.assertEqual(list(after), list(before))
        self.assertEqual(list(after), ['time_control', 'domains', 'physics', 'fdda', 'dynamics',
                                       'bdy_control', 'grib2', 'namelist_quilt', 'dfi_control'])
        for group in ('fdda', 'grib2', 'namelist_quilt'):
            self.assertEqual(after[group], {})
        for group in ('physics', 'dynamics', 'bdy_control', 'dfi_control'):
            self.assertEqual(after[group], before[group], 'physics changed in ' + group)
        registry = Path('/home/u5727520/wrf-stack/wrf/intel-adios2-dmsm/WRF/Registry')
        files = [registry / 'Registry.EM_COMMON', registry / 'registry.io_boilerplate']
        membership = {}
        for path in files:
            for line in path.read_text().splitlines():
                match = re.match(r'\s*rconfig\s+\S+\s+(\w+)\s+namelist,(\w+)\b', line)
                if match:
                    membership[match[1].lower()] = match[2].lower()
        # Auxiliary stream declarations are generated by Registry; validate the
        # frozen generated Fortran NAMELIST declarations as an independent source.
        declarations = registry.parent / 'inc/namelist_statements.inc'
        files.append(declarations)
        for line in declarations.read_text().splitlines():
            match = re.match(r'\s*NAMELIST\s+/(\w+)/\s+(\w+)\s*$', line, re.I)
            if match:
                key, group = match[2].lower(), match[1].lower()
                if key in membership:
                    self.assertEqual(membership[key], group, 'Registry/generated declaration mismatch')
                membership[key] = group
        for group, values in after.items():
            for key in values:
                self.assertTrue(key in membership, 'missing frozen Registry declaration: ' + key)
                self.assertEqual(membership[key], group, 'wrong native namelist group: ' + key)
        proof = {'original_sha256': hashlib.sha256(original.encode()).hexdigest(),
                 'generated_sha256': hashlib.sha256(generated.encode()).hexdigest(),
                 'registry_sha256': {str(path): prep.sha256(path) for path in files},
                 'groups': {group: list(values) for group, values in after.items()}}
        prep.write_json(self.root / 'actual-registry-membership.json', proof)

    def test_symlink_traversal_missing_duplicate_size_and_changed_source(self):
        self.make_archive(altered='wrfbdy_d01')
        with self.assertRaisesRegex(ValueError, 'nonregular'):
            prep.make_plan(self.archive, self.output, self.expected)
        self.make_archive(extra='../escape')
        with self.assertRaisesRegex(ValueError, 'unsafe'):
            prep.make_plan(self.archive, self.output, self.expected)
        self.make_archive(extra=prep.PREFIX + 'wrfbdy_d01')
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            prep.make_plan(self.archive, self.output, self.expected)
        self.make_archive()
        with self.assertRaisesRegex(ValueError, 'size'):
            prep.make_plan(self.archive, self.output, dict(self.expected, wrfbdy_d01=1))
        plan = prep.make_plan(self.archive, self.output, self.expected)
        with self.archive.open('r+b') as stream:
            stream.seek(plan['members']['wrfbdy_d01']['offset'])
            stream.write(b'X')
        with self.assertRaisesRegex(ValueError, 'changed'):
            prep.extract(plan, self.expected)
        self.assertFalse(self.output.exists())
        del self.data['wrfbdy_d01']
        self.make_archive()
        with self.assertRaisesRegex(ValueError, 'missing'):
            prep.make_plan(self.archive, self.output, self.expected)


if __name__ == '__main__':
    unittest.main()
