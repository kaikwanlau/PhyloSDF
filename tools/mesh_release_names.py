#!/usr/bin/env python3
"""Render, download or apply the reviewed GMM/PhyloSDF mesh names."""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import time
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

REPO = 'kaikwanlau/PhyloSDF'
ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'meshes/names.csv'
GROUPS = {'GMM': 'GMM outputs', 'generated': 'PhyloSDF outputs'}
METHODS = {
    'GMM': {'tree_per_species_gaussian', 'historical_GMM', 'per_species_shrinkage_gaussian',
            'actual_em_diagonal_gmm', 'legacy_GMM'},
    'generated': {'tree_pooled_residual_flow', 'pooled_residual_CFM'},
}


class SafeRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        result = super().redirect_request(req, fp, code, msg, headers, newurl)
        if result is not None and urlparse(newurl).netloc != 'api.github.com':
            result.remove_header('Authorization')
        return result


def token():
    return os.environ.get('GH_TOKEN') or os.environ.get('GITHUB_TOKEN')


def load_plan(path=PLAN):
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    paths, ids, release_names = set(), set(), set()
    for row in rows:
        assert row['folder'] in GROUPS, 'Unknown mesh group'
        assert row['method'] in METHODS[row['folder']], 'Wrong model group'
        assert re.fullmatch(r'[A-Za-z0-9_]+\.stl', row['filename']), 'Unsafe filename'
        assert row['path'] == row['folder'] + '/' + row['filename'], 'Invalid mesh path'
        assert re.fullmatch(r'[0-9a-f]{64}', row['sha256']), 'Invalid checksum'
        assert row['asset_id'].isdigit() and int(row['size_bytes']) > 0, 'Invalid asset identity'
        assert row['path'] not in paths and row['asset_id'] not in ids, 'Duplicate mesh identity'
        identity = row['release_tag'], row['filename']
        assert identity not in release_names, 'Release filename collision'
        paths.add(row['path']); ids.add(row['asset_id']); release_names.add(identity)
        if row['folder'] == 'generated':
            assert row['species'] and row['seed'] in ('42', '43', '44'), 'Missing generation metadata'
    return rows


def csv_text(rows):
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output, list(rows[0]), lineterminator='\n')
    writer.writeheader(); writer.writerows(rows)
    return output.getvalue()


def render(rows, root=ROOT):
    """Render only the two requested catalogue folders; preserve the full CSV."""
    result = {'meshes/names.csv': csv_text(rows)}
    pending = any(r['download_url'] != r['target_download_url'] for r in rows)
    counts = Counter(r['folder'] for r in rows)
    overview = [
        '# Mesh downloads', '',
        '| Folder | Contents | STL files |', '| :--- | :--- | ---: |',
        f"| [GMM](GMM/README.md) | GMM and per-species Gaussian baseline outputs | {counts['GMM']:,} |",
        f"| [generated](generated/README.md) | PhyloSDF pooled residual-flow outputs only | {counts['generated']:,} |", '',
        'The complete archive remains on GitHub: **7,305 STL files, 65.90 GB, across 12 releases**. '
        'The two folders above feature GMM and PhyloSDF outputs. Other baselines, controls, interpolations '
        'and reference meshes remain available through the [full archive CSV](catalogue.csv) and '
        '[GitHub releases](https://github.com/kaikwanlau/PhyloSDF/releases). No mesh is deleted.', '',
        '## Filenames', '',
        '- PhyloSDF: `generated/G_fortis_seed42_001.stl`.',
        '- GMM: `GMM/GMM_G_fortis_seed42_001.stl` when species and seed are recorded.',
        '- Historical GMM files omit an unrecorded seed; mixtures without a species label use `GMM_mixture_001.stl`.', '',
        'The number uniquely identifies a file within its group, species and seed. It is not the original '
        'generation slot. The name and provenance index preserves the original filename, source '
        'path, experiment, sampling mode, original slot, release asset ID and SHA-256 checksum.', '',
        'A species in a species-mixture filename identifies the centroid selected by the recorded generation '
        'schedule. The original conditional/mixture distinction is retained in the index.', '',
        '## Download into the two folders', '',
        'From the repository root:', '', '```bash',
        'python tools/mesh_release_names.py download --folder generated --species G.Fortis --seed 42 --limit 3',
        'python tools/mesh_release_names.py download --folder GMM --limit 3', '```', '',
        'Files are saved under `downloaded_meshes/generated/` or `downloaded_meshes/GMM/` using the short names. '
        'Each download is checked against its size and SHA-256. Omit `--limit` to download all matching files. '
        'Private-repository downloads use `GH_TOKEN` or `GITHUB_TOKEN` from your local environment.', '',
        'You can also use individual STL links on the catalogue pages. The STL data stays in GitHub Releases; '
        'the repository contains the catalogue and compact examples.', '',
    ]
    if pending:
        overview += ['The short names are ready in this catalogue. Until the release-asset migration is applied, '
                     'direct STL links may save the original long filename; the command above saves the short filename.', '']
    overview += ['## Coordinates and source data', '',
        'STL is unitless. Renaming does not repair, rescale or otherwise change geometry. Consult the source '
        'manifests and experiment fields for coordinates and extraction settings. The '
        '[100 original prepared inputs](../data/source_meshes/README.md) remain in `data/source_meshes/` with '
        'their original attribution and data terms. See the [compact output notes](../outputs/README.md) for '
        'figure and anatomical examples.', '']
    result['meshes/README.md'] = '\n'.join(overview)
    for folder, title in GROUPS.items():
        selected = sorted((r for r in rows if r['folder'] == folder), key=lambda r: r['filename'])
        pages = [selected[i:i + 200] for i in range(0, len(selected), 200)]
        index = [f'# {title}', '', '[All mesh downloads](../README.md)', '', f'**{len(selected):,} STL files**', '']
        if folder == 'generated':
            index += ['This folder contains pooled residual-flow outputs: the dated-tree comparison and the '
                      'primary-collection best/final decoder outputs. GMM, empirical residual sampling, pooled '
                      'Gaussian controls, other flow variants, historical conditional CFM and interpolations '
                      'are kept in the full archive rather than presented as PhyloSDF final-model outputs.', '']
        else:
            index += ['This folder contains the dated-tree per-species Gaussian baseline, historical GMM outputs, '
                      'the matched per-species shrinkage Gaussian, the EM diagonal mixture and legacy GMM outputs. '
                      'An unrecorded seed is left blank rather than assigned a new value.', '']
        index += ['| Experiment | Files |', '| :--- | ---: |']
        index += [f'| {name} | {count:,} |' for name, count in sorted(Counter(r['experiment'] for r in selected).items())]
        index += ['', '| Catalogue | Files |', '| :--- | ---: |']
        for i, page in enumerate(pages, 1):
            name = f'catalogue-{i:02d}.md'
            index.append(f'| [Page {i}]({name}) | {(i-1)*200+1}–{(i-1)*200+len(page)} |')
            lines = [f'# {title} — page {i}', '', '[Folder index](README.md) · [All mesh downloads](../README.md)', '',
                     '| Mesh | Species | Seed | Experiment | Size | Download |',
                     '| :--- | :--- | ---: | :--- | ---: | :--- |']
            for row in page:
                size = int(row['size_bytes']) / 1_000_000
                lines.append(f"| `{row['filename']}` | {row['species'] or 'mixture / ancestor'} | {row['seed'] or '—'} | "
                             f"{row['experiment']} | {size:.2f} MB | [STL]({row['download_url']}) |")
            result[f'meshes/{folder}/{name}'] = '\n'.join(lines) + '\n'
        index += ['']
        result[f'meshes/{folder}/README.md'] = '\n'.join(index)
    for path, content in result.items():
        target = root / path; target.parent.mkdir(parents=True, exist_ok=True); target.write_text(content)
    return result


class GitHub:
    def __init__(self):
        if not token():
            raise RuntimeError('Set GH_TOKEN or GITHUB_TOKEN locally to access private release assets.')
        self.opener = build_opener(SafeRedirect())
        self.remaining, self.reset = 1000, 0

    def call(self, path, data=None, method=None):
        for attempt in range(8):
            if self.remaining < 12:
                seconds = max(1, self.reset - time.time() + 2)
                print(f'Waiting {seconds:.0f}s for GitHub rate-limit reset.', flush=True)
                time.sleep(seconds)
            headers = {'Authorization': 'Bearer ' + token(), 'Accept': 'application/vnd.github+json', 'Content-Type': 'application/json',
                       'X-GitHub-Api-Version': '2026-03-10'}
            body = json.dumps(data).encode() if data is not None else None
            request = Request('https://api.github.com/repos/' + REPO + '/' + path,
                              data=body, headers=headers, method=method)
            try:
                with self.opener.open(request, timeout=60) as response:
                    self.remaining = int(response.headers.get('X-RateLimit-Remaining', '1000'))
                    self.reset = int(response.headers.get('X-RateLimit-Reset', '0'))
                    return json.load(response)
            except HTTPError as error:
                if error.code not in (403, 429):
                    raise
                remaining = error.headers.get('X-RateLimit-Remaining')
                retry = error.headers.get('Retry-After')
                detail = error.read().decode(errors='replace')
                if remaining != '0' and not retry and 'rate limit' not in detail.lower():
                    raise RuntimeError(f'GitHub refused {path}: HTTP {error.code}') from error
                seconds = float(retry) if retry else (max(1, int(error.headers.get('X-RateLimit-Reset', '0')) - time.time() + 2)
                                                      if remaining == '0' else min(900, 60 * 2**attempt))
                print(f'GitHub requested a pause; waiting {seconds:.0f}s.', flush=True)
                time.sleep(seconds)
        raise RuntimeError('GitHub rate-limit retries exhausted; rerun the resumable migration.')

    def assets(self, rows):
        found = {}
        for tag in sorted({r['release_tag'] for r in rows}):
            release = self.call('releases/tags/' + tag)
            if release.get('immutable'):
                raise RuntimeError('Release is immutable: ' + tag)
            cached = release.get('assets', [])
            found.update({str(a['id']): a for a in cached})
            wanted = {r['asset_id'] for r in rows if r['release_tag'] == tag}
            if wanted <= set(found):
                continue
            for page in range(1, 100):
                assets = self.call(f"releases/{release['id']}/assets?per_page=100&page={page}")
                found.update({str(a['id']): a for a in assets})
                if len(assets) < 100:
                    break
        return found


def preflight(rows, assets):
    for row in rows:
        asset = assets.get(row['asset_id'])
        if not asset or asset['name'] not in (row['original_filename'], row['filename']):
            raise RuntimeError('Asset identity changed: ' + row['path'])
        if asset['size'] != int(row['size_bytes']) or asset.get('digest') != 'sha256:' + row['sha256']:
            raise RuntimeError('Asset content changed: ' + row['path'])
        for other in assets.values():
            if other['id'] != asset['id'] and other['name'] == row['filename']:
                # Release tags are encoded in the download URL; names need only be unique within a release.
                expected = '/releases/download/' + row['release_tag'] + '/'
                if expected in other['browser_download_url']:
                    raise RuntimeError('Existing asset name collision: ' + row['filename'])


def migrate(rows, apply=False, publish_branch=None, max_renames=None):
    github = GitHub()
    assets = github.assets(rows)
    preflight(rows, assets)
    todo = [r for r in rows if assets[r['asset_id']]['name'] != r['filename']]
    print(json.dumps({'validated': len(rows), 'remaining_renames': len(todo), 'apply': apply}), flush=True)
    if not apply:
        return
    if max_renames is not None:
        todo = todo[:max_renames]
    for i, row in enumerate(todo, 1):
        changed = github.call('releases/assets/' + row['asset_id'], {'name': row['filename']}, 'PATCH')
        if changed['size'] != int(row['size_bytes']) or changed.get('digest') != 'sha256:' + row['sha256']:
            raise RuntimeError('Renamed asset content mismatch: ' + row['path'])
        assets[row['asset_id']] = changed
        if i % 25 == 0 or i == len(todo):
            print(f'Renamed {i}/{len(todo)} assets; mesh contents are unchanged.', flush=True)
        # Stay below GitHub's documented content-creation request budget.
        time.sleep(7.5)
    for row in rows:
        row['download_url'] = assets[row['asset_id']]['browser_download_url']
        assert assets[row['asset_id']]['name'] in (row['original_filename'], row['filename'])
    files = render(rows)
    # Refresh the complete archive index by immutable asset ID / original filename.
    full = ROOT / 'meshes/catalogue.csv'
    if full.is_file():
        with full.open(newline='') as stream:
            archive = list(csv.DictReader(stream))
        changes = {(r['release_tag'], r['original_filename']): r for r in rows}
        for record in archive:
            row = changes.get((record['release_tag'], record['filename']))
            if row:
                record['filename'], record['download_url'] = row['filename'], row['download_url']
        files['meshes/catalogue.csv'] = csv_text(archive)
        full.write_text(files['meshes/catalogue.csv'])
    if publish_branch:
        head = github.call('git/ref/heads/' + publish_branch)['object']['sha']
        tree = github.call('git/trees/' + head + '?recursive=1')
        current = next(f['sha'] for f in tree['tree'] if f['path'] == 'meshes/names.csv')
        latest = base64.b64decode(github.call('git/blobs/' + current)['content']).decode()
        identity = lambda data: [{k: v for k, v in r.items() if k != 'download_url'} for r in csv.DictReader(io.StringIO(data))]
        if identity(latest) != identity(csv_text(rows)):
            raise RuntimeError('The reviewed name plan changed during migration; catalogue publication stopped.')
        new_tree = github.call('git/trees', {'base_tree': tree['sha'], 'tree': [
            {'path': path, 'mode': '100644', 'type': 'blob', 'content': content} for path, content in files.items()]}, 'POST')
        commit = github.call('git/commits', {'message': 'Refresh mesh links after release filename migration',
            'tree': new_tree['sha'], 'parents': [head]}, 'POST')
        github.call('git/refs/heads/' + publish_branch, {'sha': commit['sha'], 'force': False}, 'PATCH')
        print('Published catalogue links: ' + commit['sha'], flush=True)


def download(rows, args):
    selected = [r for r in rows if r['folder'] == args.folder and
                (not args.species or r['species'].lower() == args.species.lower()) and
                (args.seed is None or r['seed'] == str(args.seed))]
    if args.limit is not None:
        selected = selected[:args.limit]
    if not selected:
        raise RuntimeError('No meshes match these filters.')
    opener = build_opener(SafeRedirect())
    for row in selected:
        destination = args.output / row['path']
        if destination.exists():
            if hashlib.sha256(destination.read_bytes()).hexdigest() == row['sha256']:
                print('Verified existing ' + str(destination)); continue
            raise RuntimeError('Existing file has different contents: ' + str(destination))
        if token():
            url = 'https://api.github.com/repos/' + REPO + '/releases/assets/' + row['asset_id']
            headers = {'Accept': 'application/octet-stream', 'Authorization': 'Bearer ' + token()}
        else:
            url, headers = row['download_url'], {}
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix('.stl.part')
        digest, size = hashlib.sha256(), 0
        try:
            with opener.open(Request(url, headers=headers), timeout=120) as response, temporary.open('wb') as stream:
                while chunk := response.read(1024 * 1024):
                    digest.update(chunk); size += len(chunk); stream.write(chunk)
            if size != int(row['size_bytes']) or digest.hexdigest() != row['sha256']:
                raise RuntimeError('Downloaded mesh checksum mismatch: ' + row['path'])
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        print('Downloaded and verified ' + str(destination))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('validate')
    commands.add_parser('render')
    fetch = commands.add_parser('download')
    fetch.add_argument('--folder', choices=GROUPS, required=True)
    fetch.add_argument('--species')
    fetch.add_argument('--seed', type=int)
    fetch.add_argument('--limit', type=int)
    fetch.add_argument('--output', type=Path, default=Path('downloaded_meshes'))
    migration = commands.add_parser('migrate')
    migration.add_argument('--apply', action='store_true', help='Rename the reviewed GitHub release assets')
    migration.add_argument('--publish-branch', help='Publish refreshed catalogue links after all renames succeed')
    migration.add_argument('--max-renames', type=int, help='Limit one resumable batch; remaining assets retain their current links')
    args = parser.parse_args(argv)
    rows = load_plan()
    if args.command == 'render':
        render(rows)
    elif args.command == 'download':
        if args.limit is not None and args.limit < 1:
            parser.error('--limit must be positive')
        download(rows, args)
    elif args.command == 'migrate':
        if args.publish_branch and not args.apply:
            parser.error('--publish-branch requires --apply')
        if args.max_renames is not None and args.max_renames < 1:
            parser.error('--max-renames must be positive')
        migrate(rows, args.apply, args.publish_branch, args.max_renames)
    else:
        print(json.dumps({'files': len(rows), 'folders': dict(Counter(r['folder'] for r in rows))}))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (AssertionError, HTTPError, RuntimeError) as error:
        print('Error: ' + str(error), file=sys.stderr)
        raise SystemExit(1)
