"""Safety checks for mesh identity, data preservation and named downloads."""
import copy
import hashlib
import io
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
from urllib.request import Request

import mesh_release_names as mesh


class MeshSafetyTests(unittest.TestCase):
    def setUp(self):
        self.row = copy.deepcopy(mesh.load_plan()[0])
        self.asset = {'id': int(self.row['asset_id']), 'name': self.row['original_filename'],
                      'size': int(self.row['size_bytes']), 'digest': 'sha256:' + self.row['sha256'],
                      'browser_download_url': self.row['download_url']}

    def test_content_identity_required_before_rename(self):
        mesh.preflight([self.row], {self.row['asset_id']: self.asset})
        self.asset['digest'] = 'sha256:' + '0' * 64
        with self.assertRaisesRegex(RuntimeError, 'content changed'):
            mesh.preflight([self.row], {self.row['asset_id']: self.asset})

    def test_partial_migration_can_be_resumed_by_asset_id(self):
        self.asset['name'] = self.row['filename']
        self.asset['browser_download_url'] = self.row['target_download_url']
        mesh.preflight([self.row], {self.row['asset_id']: self.asset})

    def test_existing_release_filename_is_not_overwritten(self):
        other = copy.deepcopy(self.asset)
        other['id'] += 1
        other['name'] = self.row['filename']
        other['browser_download_url'] = self.row['target_download_url']
        with self.assertRaisesRegex(RuntimeError, 'collision'):
            mesh.preflight([self.row], {self.row['asset_id']: self.asset, str(other['id']): other})

    def test_download_saves_short_name_and_checks_original_bytes(self):
        payload = b'original mesh bytes'
        self.row['size_bytes'] = str(len(payload))
        self.row['sha256'] = hashlib.sha256(payload).hexdigest()
        opener = SimpleNamespace(open=lambda *a, **kw: io.BytesIO(payload))
        with tempfile.TemporaryDirectory() as temporary, patch.object(mesh, 'token', return_value=None), patch.object(mesh, 'build_opener', return_value=opener):
            args = SimpleNamespace(folder=self.row['folder'], species=None, seed=None, limit=1, output=Path(temporary))
            mesh.download([self.row], args)
            target = Path(temporary) / self.row['path']
            self.assertEqual(target.read_bytes(), payload)
            target.unlink()
            self.row['sha256'] = '0' * 64
            with self.assertRaisesRegex(RuntimeError, 'checksum mismatch'):
                mesh.download([self.row], args)
            self.assertFalse(target.exists())
            self.assertFalse(target.with_suffix('.stl.part').exists())

    def test_download_does_not_overwrite_an_existing_different_mesh(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / self.row['path']
            target.parent.mkdir(parents=True)
            target.write_bytes(b'keep this file')
            args = SimpleNamespace(folder=self.row['folder'], species=None, seed=None, limit=1, output=Path(temporary))
            with self.assertRaisesRegex(RuntimeError, 'different contents'):
                mesh.download([self.row], args)
            self.assertEqual(target.read_bytes(), b'keep this file')

    def test_download_redirect_does_not_send_github_credentials_to_storage(self):
        req = Request('https://api.github.com/repos/kaikwanlau/PhyloSDF/releases/assets/1',
                      headers={'Authorization': 'Bearer test-placeholder'})
        redirect = mesh.SafeRedirect().redirect_request(req, None, 302, 'Found', {},
                                                      'https://release-assets.githubusercontent.com/test')
        self.assertIsNone(redirect.get_header('Authorization'))

    def test_limited_migration_preserves_links_for_unrenamed_assets(self):
        second = copy.deepcopy(self.row)
        second['asset_id'] = str(int(second['asset_id']) + 1)
        second['filename'] = second['filename'].replace('_001.stl', '_002.stl')
        second['path'] = second['folder'] + '/' + second['filename']
        second['original_filename'] = 'other_original_mesh.stl'
        second['download_url'] = self.row['download_url'].rsplit('/', 1)[0] + '/' + second['original_filename']
        second['target_download_url'] = self.row['target_download_url'].rsplit('/', 1)[0] + '/' + second['filename']
        assets = {self.row['asset_id']: copy.deepcopy(self.asset), second['asset_id']: {
            **copy.deepcopy(self.asset), 'id': int(second['asset_id']), 'name': second['original_filename'],
            'browser_download_url': second['download_url']}}
        calls = []
        def rename(path, data, method):
            calls.append(path)
            asset = assets[path.rsplit('/', 1)[1]]
            asset['name'] = data['name']
            asset['browser_download_url'] = self.row['target_download_url']
            return asset
        fake = SimpleNamespace(assets=lambda rows: assets, call=rename)
        with tempfile.TemporaryDirectory() as temporary, patch.object(mesh, 'ROOT', Path(temporary)), \
             patch.object(mesh, 'GitHub', return_value=fake), patch.object(mesh.time, 'sleep'), \
             patch.object(mesh, 'render', return_value={}):
            rows = [copy.deepcopy(self.row), copy.deepcopy(second)]
            mesh.migrate(rows, apply=True, max_renames=1)
            self.assertEqual(len(calls), 1)
            self.assertEqual(rows[0]['download_url'], self.row['target_download_url'])
            self.assertEqual(rows[1]['download_url'], second['download_url'])


if __name__ == '__main__':
    unittest.main()
