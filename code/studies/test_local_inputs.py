"""Check local input routing without exposing private storage identifiers."""
import importlib.util
import json
from pathlib import Path
import re
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('local_study_launcher', HERE/'run.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class LocalReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.args = SimpleNamespace(study='final', workspace=self.root/'work',
                                    inputs=self.root/'inputs', references=self.root/'references')
        self.engine = SimpleNamespace(INPUTS={})
        with patch.object(launcher.importlib, 'import_module', return_value=self.engine):
            launcher.load_engine(self.args)
        row = json.loads(launcher.REFERENCE_MANIFEST.read_text())['references'][0]
        self.reference = row
        self.source = self.args.references/row['raw_stl']
        self.source.parent.mkdir(parents=True)
        self.source.write_bytes(b'local reference fixture')
        self.target = self.engine.REFS/row['raw_stl']

    def test_published_local_alias_resolves(self):
        self.engine.fetch(self.reference['raw_drive_id'], self.target)
        self.assertEqual(self.target.read_bytes(), self.source.read_bytes())

    def test_private_archive_identifier_is_not_needed_for_an_approved_path(self):
        self.engine.fetch('opaque-archive-reference', self.target)
        self.assertEqual(self.target.read_bytes(), self.source.read_bytes())

    def test_unknown_destination_is_rejected(self):
        unknown = self.args.references/'raw_stl/not-a-reference.stl'
        unknown.write_bytes(b'not an approved specimen')
        target = self.engine.REFS/'raw_stl/not-a-reference.stl'
        with self.assertRaises(FileNotFoundError):
            self.engine.fetch('opaque-archive-reference', target)
        self.assertFalse(target.exists())

    def test_destination_outside_reference_root_is_rejected(self):
        target = self.root/'outside.stl'
        with self.assertRaises(FileNotFoundError):
            self.engine.fetch('opaque-archive-reference', target)
        self.assertFalse(target.exists())

    def test_automatic_remote_access_remains_disabled(self):
        with self.assertRaisesRegex(RuntimeError, 'local input files only'):
            self.engine.drive()

    def test_published_reference_locations_are_local_aliases(self):
        manifest = json.loads(launcher.REFERENCE_MANIFEST.read_text())
        for row in manifest['references']:
            self.assertTrue(row['raw_drive_id'].startswith('local-input-'))
            self.assertTrue(row['processed_drive_id'].startswith('local-input-'))
        for row in manifest['files']:
            self.assertIsNone(row['url'])
            self.assertIsNone(row['streamed_reference_id'])

    def test_archive_specs_do_not_publish_personal_storage_ids(self):
        for study in launcher.STUDIES:
            for ids, checksum in launcher.input_specs(study).values():
                identifiers = ids if isinstance(ids, (list, tuple)) else [ids]
                self.assertTrue(all(value.startswith('local-input-') for value in identifiers))

    def test_current_text_files_do_not_publish_drive_urls(self):
        root = HERE.parents[1]
        pattern = re.compile(r'https?://(?:drive|docs)\.google\.com(?:/|\b)', re.I)
        for path in root.rglob('*'):
            if not path.is_file() or '.git' in path.parts or path.suffix not in {'.py','.json','.md','.csv','.yml','.yaml'}:
                continue
            self.assertIsNone(pattern.search(path.read_text()), str(path.relative_to(root)))


if __name__ == '__main__':
    unittest.main()
