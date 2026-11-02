# fm_engine.py (v2) -- PhyloSDF final-model ablation under the matched fixed-update protocol.
#
# v2 changes (same science, same numbers): the Chamfer nearest-neighbour search uses the frozen evaluator's exact
# KD-tree again (multithreaded) instead of a GPU kernel that was very slow on L4/A100; a progress zip is saved after
# every single run; an uploaded progress zip lets a new session continue where a disconnected one stopped; seed 42 of
# every arm runs first, so even a stopped run leaves one complete ablation table.
#
# Google Drive is used READ-ONLY: this code only lists/downloads files by ID. It never uploads,
# edits, moves, deletes or shares anything. All outputs stay in /content and are downloaded
# to your computer as zip files.
#
# What it runs (final decoder, 10,000 updates, identical initial weights and sampling inputs
# as the six-arm study):
#   pooled_residual        FINAL MODEL: residual flow pooled across species (neutral conditioning),
#                          real codes only, variance term 0.1, guidance 1, temperature 1
#   species_fullcode       direct generation: species-conditioned flow on full codes, real codes only
#   species_residual       residual flow conditioned on species (not pooled), real codes only
#   pooled_residual_no_var final model without the variance term
#   pooled_residual_aug    final model trained with the original latent augmentation
#   pooled_residual_tcal   final model, sampling temperature calibrated on the TRAINING residual variance
# Each arm: 140 codes (100 species-mixture + 4 focal species x 10) -> 256^3 meshes -> frozen metrics.
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys, json, time, hashlib, zipfile, shutil, importlib.util, importlib.metadata, datetime
from pathlib import Path
import numpy as np

ROOT = Path('/content/FM')
DL = ROOT / 'downloads'
SRC = ROOT / 'src' / 'matched_cfm'
MESHSRC = ROOT / 'meshsrc' / 'new_mesh_metrics'
REFS = ROOT / 'refs'
WORK = ROOT / 'work'
OUT = ROOT / 'results'
SURFACE_SEEDS = (20261014, 20261015)
FOCAL = ('G.Fortis', 'G.Magnirostris', 'G.Fuliginosa', 'C.Parvulus')
FINAL_DECODER_SHA = 'd73acbdd6bd2836a352ad3e7434699292577492f813c9f305769115362cc69c7'
# Real within-species diversity of the focal species (mean of both surface seeds), for display only.
REAL_WITHIN = {'G.Fortis': 0.007346849200893287, 'G.Magnirostris': 0.006132724795045568,
               'G.Fuliginosa': 0.007040538559969167, 'C.Parvulus': 0.009734557443890233}
REAL_MIXTURE_DIVERSITY = 0.011483656538536259   # species-balanced, real skulls
REAL_LOO_NN = 0.0014668375205922                 # real skull to nearest other real skull

INPUTS = {
    'matched_cfm_source_and_inputs_20261018_v1.zip': (
        ['local-input-4547bb9fdfe0aaff', 'local-input-ec3acf21ef60d480'],
        '93f2dc31651f900981076010a6a039bd84314434c41270fbdf64e97551cef169'),
    'prepared_final_seed42.zip': (
        ['local-input-e45c578d3b69d709', 'local-input-fb282e2f87c0858f'],
        '9ef97b2a0833c18743d724b441233f1a33140982fc51c81a46d1075e0b4144b6'),
    'new_matched_mesh800_source_20261018_v1.zip': (
        ['local-input-84ff5aba929ef1c4', 'local-input-b634c9ff5803ca39'],
        '30e17ce1efa4a478e5253b29f9885f4b2f3999a1b1ea27599550dc2220577eb7'),
    'full10000_no_augmentation_generation140.zip': (
        ['local-input-75cfdf8d7a09afce'],
        '09c95f4dd1cf2aad9e2cf58ac60256612d0b7d0172e01b9069873ca12101509c'),
}

# New predeclared variants (added next to the six original ones; the original code is not edited).
NEW_VARIANTS = {
    'fm_pooled_residual': {'augmentation': False, 'phylo_conditioning': False, 'target': 'residual',
                           'var_weight': .1, 'neutral': True},
    'fm_pooled_residual_no_var': {'augmentation': False, 'phylo_conditioning': False, 'target': 'residual',
                                  'var_weight': 0., 'neutral': True},
    'fm_pooled_residual_aug': {'augmentation': True, 'phylo_conditioning': False, 'target': 'residual',
                               'var_weight': .1, 'neutral': True},
    'fm_species_fullcode': {'augmentation': False, 'phylo_conditioning': True, 'target': 'full_code',
                            'var_weight': 0.},
}
ARMS = {
    'pooled_residual': dict(variant='fm_pooled_residual', neutral=True, guidance=1.0, temperature=1.0,
                            method='fm_pooled_residual_cfm'),
    'species_fullcode': dict(variant='fm_species_fullcode', neutral=False, guidance=1.0, temperature=1.0,
                             method='fm_species_fullcode_cfm_g1'),
    'species_residual': dict(variant='no_augmentation', neutral=False, guidance=1.0, temperature=1.0,
                             method='fm_species_residual_cfm_g1'),
    'pooled_residual_no_var': dict(variant='fm_pooled_residual_no_var', neutral=True, guidance=1.0, temperature=1.0,
                                   method='fm_pooled_residual_no_var'),
    'pooled_residual_aug': dict(variant='fm_pooled_residual_aug', neutral=True, guidance=1.0, temperature=1.0,
                                method='fm_pooled_residual_aug'),
    'pooled_residual_tcal': dict(variant='fm_pooled_residual', neutral=True, guidance=1.0, temperature='calibrated',
                                 method='fm_pooled_residual_tcal'),
}
STUDY_GROUP, STUDY_ARM = 'final_model_ablation_v1', 'matched_fixed_protocol'
HKT = datetime.timezone(datetime.timedelta(hours=8))


def log(msg):
    print(f"[{datetime.datetime.now(HKT):%m-%d %H:%M:%S} HKT] {msg}", flush=True)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=1, default=float))
    tmp.replace(path)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- Drive (read-only) ----------
_svc = None


def drive():
    global _svc
    if _svc is None:
        from google.colab import auth
        auth.authenticate_user()
        import google.auth
        from googleapiclient.discovery import build
        _svc = build('drive', 'v3', credentials=google.auth.default()[0], cache_discovery=False)
    return _svc


def fetch(file_id, path):
    from googleapiclient.http import MediaIoBaseDownload
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.part')
    with open(tmp, 'wb') as fh:
        request = drive().files().get_media(fileId=file_id, supportsAllDrives=True)
        downloader = MediaIoBaseDownload(fh, request, chunksize=32 << 20)
        done = False
        while not done:
            _, done = downloader.next_chunk(num_retries=5)
    tmp.replace(path)


def get_input(name):
    ids, digest = INPUTS[name]
    path = DL / name
    if path.exists() and sha(path) == digest:
        return path
    problem = None
    for file_id in ids:
        try:
            fetch(file_id, path)
            if sha(path) == digest:
                return path
            problem = f'checksum mismatch from {file_id}'
        except Exception as error:  # try the backup copy
            problem = repr(error)
    raise RuntimeError(f'Could not download {name}: {problem}')


def setup():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('No GPU. Runtime > Change runtime type > GPU (L4 or A100), then run again.')
    for d in (DL, WORK, OUT):
        d.mkdir(parents=True, exist_ok=True)
    log('downloading inputs (read-only)...')
    with zipfile.ZipFile(get_input('matched_cfm_source_and_inputs_20261018_v1.zip')) as z:
        if not (SRC / 'run_matched_cfm.py').exists():
            z.extractall(ROOT / 'src')
    with zipfile.ZipFile(get_input('new_matched_mesh800_source_20261018_v1.zip')) as z:
        if not (MESHSRC / 'vendor' / 'vendor_metrics.py').exists():
            z.extractall(ROOT / 'meshsrc')
    with zipfile.ZipFile(get_input('prepared_final_seed42.zip')) as z:
        if not (WORK / 'final' / 'seed_42' / 'prepared' / 'PREPARED.json').exists():
            for member in z.namelist():
                if member.startswith('final/'):
                    z.extract(member, WORK)
    get_input('full10000_no_augmentation_generation140.zip')
    if sha(SRC / 'inputs' / 'final.pt') != FINAL_DECODER_SHA:
        raise RuntimeError('final.pt checksum differs from the decoder used in the six-arm study')
    manifest = MESHSRC / 'vendor' / 'reference_manifest.json'
    rows = json.loads(manifest.read_text())['references']
    log(f'downloading the {len(rows)} real reference skulls (raw STL + processed NPZ, read-only)...')
    for i, row in enumerate(rows):
        for path_key, id_key, sha_key in (('raw_stl', 'raw_drive_id', 'raw_sha256'),
                                          ('processed_npz', 'processed_drive_id', 'processed_sha256')):
            target = REFS / row[path_key]
            if target.exists() and sha(target) == row[sha_key]:
                continue
            fetch(row[id_key], target)
            if sha(target) != row[sha_key]:
                raise RuntimeError('reference checksum mismatch: ' + row[path_key])
        if (i + 1) % 25 == 0:
            log(f'  references {i + 1}/{len(rows)}')
    shutil.copy(manifest, REFS / 'reference_manifest.json')
    env = {'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
           'numpy': np.__version__, 'python': sys.version.split()[0]}
    for name in ('scipy', 'scikit-image', 'trimesh'):
        env[name] = importlib.metadata.version(name)
    write_json(OUT / 'ENVIRONMENT.json', env)
    log('environment: ' + json.dumps(env))


# ---------------------------------------------------------------- matched CFM code -----------
_rmc = None


def rmc():
    """The original run_matched_cfm module with four extra predeclared variants.
    Neutral variants feed the null species token, zero size and zero phylogeny to the flow,
    i.e. the residual flow is pooled across species; species enters only through the centroid."""
    global _rmc
    if _rmc is None:
        import torch
        torch.set_num_threads(2)
        module = load_module(SRC / 'run_matched_cfm.py', 'run_matched_cfm_fm')
        module.VARIANTS.update(NEW_VARIANTS)
        original = module.training_tensors

        def training_tensors(arrays, variant):
            tensors, weights = original(arrays, variant)
            if module.VARIANTS[variant].get('neutral'):
                tensors['species'] = torch.full_like(tensors['species'], 24)
                tensors['bbox'] = torch.zeros_like(tensors['bbox'])
                tensors['phylo'] = torch.zeros_like(tensors['phylo'])
            return tensors, weights

        module.training_tensors = training_tensors
        _rmc = module
    return _rmc


def prepared_path(seed):
    path = WORK / 'final' / f'seed_{seed}' / 'prepared'
    if not (path / 'PREPARED.json').exists():
        if path.exists():
            shutil.rmtree(path)
        start = time.time()
        rmc().prepare(base=str(SRC), work=str(WORK), pair='final', seed=seed, device='cuda')
        log(f'prepared seed {seed} (Hessian-guided augmentation cache) in {time.time() - start:.0f} s')
    rmc().load_prepared(path)       # verifies hashes and source closure
    return path


def trained_path(seed, variant):
    return WORK / 'final' / f'seed_{seed}' / 'arms' / variant


def train(seed, variant):
    out = trained_path(seed, variant)
    if (out / 'TRAINING_COMPLETE.json').exists():
        return out
    if out.exists():
        shutil.rmtree(out)          # an interrupted run restarts cleanly
    start = time.time()
    rmc().train(str(prepared_path(seed)), str(out), variant=variant, updates=10000, device='cuda',
                checkpoint_every=10000)
    latest = json.loads((out / 'LATEST.json').read_text())
    saved = rmc().verified_torch_load(out / latest['checkpoint'])
    trace = saved['trace']
    write_json(out / 'TRACE_SUMMARY.json', {
        'variant': variant, 'seed': seed, 'updates': len(trace),
        'every_100': [{k: r[k] for k in ('update', 'loss', 'flow_loss', 'variance_loss', 'lr')}
                      for r in trace if r['update'] % 100 == 0],
        'sampled_real': saved['sampled_real'], 'sampled_augmented': saved['sampled_augmented'],
        'seconds': time.time() - start})
    log(f'trained {variant} seed {seed} in {time.time() - start:.0f} s (final loss {trace[-1]["loss"]:.4f})')
    return out


def _flow(model, x, species, bbox, phylo, guidance, steps=50):
    import torch
    with torch.no_grad():
        for step in range(steps):
            t = torch.full((len(x),), step / float(steps), device=x.device)
            conditional = model(x, t, species, bbox, phylo)
            if guidance == 1.0:
                x += conditional / float(steps)
            else:                       # identical arithmetic to the original sampler
                unconditional = model(x, t, None, bbox, phylo)
                x += (unconditional + guidance * (conditional - unconditional)) / float(steps)
    return x


def _target_variance(arrays, cfg):
    real = np.asarray(arrays['real_codes'], dtype=np.float64)
    species = np.asarray(arrays['real_species'])
    raw = real - arrays['centroids'][species] if cfg['target'] == 'residual' else real
    target = (raw - arrays[cfg['target'] + '_mean']) / arrays[cfg['target'] + '_global_std']
    return float(np.var(target, ddof=1))


def calibrate_temperature(model, arrays, cfg, seed, n=4000):
    """Choose T so the generated (normalised) residual variance equals the TRAINING residual variance.
    Uses only training statistics and fresh noise; no evaluation data are involved."""
    import torch
    target_var = _target_variance(arrays, cfg)
    generator = torch.Generator(device='cpu').manual_seed(seed + 500000)
    noise = torch.randn((n, 64), generator=generator).to('cuda')
    species = torch.full((n,), 24, device='cuda', dtype=torch.long)
    bbox = torch.zeros((n, 3), device='cuda')
    phylo = torch.zeros((n, int(np.asarray(arrays['phylo_embeddings']).shape[1])), device='cuda')

    def variance(T):
        return float(_flow(model, noise.clone() * T, species, bbox, phylo, 1.0).var())

    v1, lo, hi = variance(1.0), 0.5, 4.0
    if variance(hi) < target_var:
        T = hi
    else:
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            if variance(mid) < target_var:
                lo = mid
            else:
                hi = mid
        T = 0.5 * (lo + hi)
    info = {'target_training_residual_variance': target_var, 'generated_variance_at_T1': v1,
            'ratio_at_T1': v1 / target_var, 'calibrated_temperature': T,
            'generated_variance_at_T': variance(T), 'calibration_samples': n,
            'calibration_noise_seed': seed + 500000, 'search_interval': [0.5, 4.0]}
    return T, info


def sample(seed, arm, out_dir, guidance=None):
    import torch
    m, A = rmc(), ARMS[arm]
    guidance = A['guidance'] if guidance is None else guidance
    prepared, trained = prepared_path(seed), trained_path(seed, A['variant'])
    if not (trained / 'TRAINING_COMPLETE.json').exists():
        raise RuntimeError('training incomplete for ' + A['variant'])
    info, arrays, initial = m.load_prepared(prepared)
    latest = json.loads((trained / 'LATEST.json').read_text())
    if m.file_sha(trained / latest['checkpoint']) != latest['sha256']:
        raise RuntimeError('checkpoint hash differs')
    saved = m.verified_torch_load(trained / latest['checkpoint'])
    cfg = m.VARIANTS[A['variant']]
    calibration = None
    with m.isolated_rng():
        model = m.load_vendor('diffusion').PhyloFlowNet(**initial['kwargs']).to('cuda')
        model.load_state_dict(saved['model'])
        model.eval()
        with np.load(prepared / 'GENERATION_INPUTS.npz', allow_pickle=False) as data:
            generation = {k: data[k] for k in data.files}
        species = torch.as_tensor(generation['species'], device='cuda')
        bbox = torch.as_tensor(generation['size_normalized'], device='cuda', dtype=torch.float32)
        phylo = torch.as_tensor(generation['phylo'], device='cuda', dtype=torch.float32)
        if not cfg['phylo_conditioning']:
            phylo = torch.zeros_like(phylo)
        if A['neutral']:
            species = torch.full_like(species, 24)
            bbox = torch.zeros_like(bbox)
            phylo = torch.zeros_like(phylo)
        T = A['temperature']
        if T == 'calibrated':
            T, calibration = calibrate_temperature(model, arrays, cfg, seed)
            log(f'calibrated temperature {T:.4f} (generated/training residual variance at T=1: '
                f'{calibration["ratio_at_T1"]:.3f})')
        x = torch.as_tensor(generation['initial_noise'], device='cuda', dtype=torch.float32).clone()
        x *= float(T)
        x = _flow(model, x, species, bbox, phylo, float(guidance))
        target = x.cpu().numpy()
    codes = target * arrays[cfg['target'] + '_global_std'] + arrays[cfg['target'] + '_mean']
    if cfg['target'] == 'residual':
        codes = codes + arrays['centroids'][generation['species']]
    codes = np.asarray(codes, dtype=np.float32)
    if codes.shape != (140, 64) or not np.isfinite(codes).all():
        raise RuntimeError('nonfinite or incomplete generated codes')
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / 'GENERATED_CODES.npz', codes=codes, normalized_targets=target,
                        species=generation['species'], cohort=generation['cohort'],
                        sample_slot=generation['sample_slot'])
    write_json(out_dir / 'GENERATION_MANIFEST.json', {
        'arm': arm, 'method': A['method'], 'variant': A['variant'], 'variant_config': cfg, 'seed': seed,
        'pair': 'final', 'guidance_scale': guidance, 'temperature': T, 'temperature_calibration': calibration,
        'euler_steps': 50, 'neutral_conditioning': A['neutral'],
        'prepared_sha256': m.file_sha(prepared / 'PREPARED.json'),
        'generation_inputs_sha256': m.file_sha(prepared / 'GENERATION_INPUTS.npz'),
        'model_checkpoint_sha256': latest['sha256'], 'codes_sha256': sha(out_dir / 'GENERATED_CODES.npz')})
    return out_dir / 'GENERATED_CODES.npz'


# ---------------------------------------------------------------- decoding (256^3) -----------
_decoder = None


def decoder():
    """The original DeepSDF decoder (final.pt) built exactly as the frozen benchmark does, float32,
    TF32 off, deterministic. Same grid, chunk size and extractor as the six-arm study."""
    global _decoder
    if _decoder is None:
        import torch
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_float32_matmul_precision('highest')
        frozen = MESHSRC / 'grounded' / 'probe_source' / 'frozen'
        backend = load_module(frozen / 'runtime_backend.py', 'fm_frozen_backend')
        bench = load_module(frozen / 'benchmark.py', 'fm_frozen_benchmark')
        source = load_module(frozen / 'deepsdf.py', 'fm_original_deepsdf')
        ck = backend.load_checkpoint(SRC / 'inputs' / 'final.pt')
        cfg = ck['config']
        keys = ['latent_dim', 'hidden_dim', 'num_layers', 'skip_connection', 'weight_norm']
        model = source.DeepSDFDecoder(**{k: cfg[k] for k in keys if k in cfg})
        model.load_state_dict({k: torch.from_numpy(np.array(v, copy=True)) for k, v in ck['decoder_state_dict'].items()})
        model.to('cuda').eval()
        points = np.random.default_rng(20261014).uniform(-1, 1, (127, 3)).astype(np.float32)
        z = ck['latent_lib_state_dict']['latent_codes.weight'][0]
        with torch.inference_mode():
            torch_values = model(torch.as_tensor(z, dtype=torch.float32, device='cuda').expand(127, -1),
                                 torch.as_tensor(points, device='cuda')).reshape(-1).cpu().numpy()
        parity = float(np.max(np.abs(torch_values - backend.NumpyDecoder(ck).predict(points, z))))
        if parity > 3e-5:
            raise RuntimeError(f'NumPy/torch decoder parity failed: {parity}')
        _decoder = (model, bench, parity)
    return _decoder


def sdf_field(code, resolution=256, chunk=8192):
    import torch
    model = decoder()[0]
    axis = torch.from_numpy(np.linspace(-1, 1, resolution, dtype=np.float32)).to('cuda')
    total = resolution ** 3
    out = torch.empty(total, dtype=torch.float32, device='cuda')
    z = torch.as_tensor(np.asarray(code, dtype=np.float32), device='cuda')
    with torch.inference_mode():
        for low in range(0, total, chunk):
            high = min(low + chunk, total)
            index = torch.arange(low, high, device='cuda')
            points = torch.stack((axis[index // (resolution * resolution)],
                                  axis[(index // resolution) % resolution],
                                  axis[index % resolution]), dim=1)
            out[low:high] = model(z.expand(high - low, -1), points).reshape(-1)
    return out.cpu().numpy().reshape(resolution, resolution, resolution)


def species_names():
    meta = json.loads((SRC / 'inputs' / 'metadata.json').read_text())
    return [meta['idx_to_species'][str(i)] for i in range(24)]


def decode_arm(seed, arm, codes_path, root, rows=None):
    bench = decoder()[1]
    method, names = ARMS[arm]['method'], species_names()
    root = Path(root)
    if root.exists():
        shutil.rmtree(root)
    (root / 'tasks').mkdir(parents=True)
    data = np.load(codes_path, allow_pickle=False)
    codes = data['codes'].astype(np.float32)
    records, start = [], time.time()
    rows = list(range(len(codes))) if rows is None else list(rows)
    for done, row in enumerate(rows, 1):
        cohort, slot, sp = str(data['cohort'][row]), int(data['sample_slot'][row]), int(data['species'][row])
        task = f'fm__{method}__seed{seed}__{cohort}__slot{slot:03d}'
        folder = root / 'tasks' / task
        folder.mkdir()
        record = {'task': task, 'method': method, 'input_role': method, 'cohort': cohort, 'sample_slot': slot,
                  'species': names[sp], 'species_idx': sp, 'group': STUDY_GROUP, 'historical_arm': STUDY_ARM,
                  'paired_reference': 'final', 'training_seed': seed, 'generation_seed': seed, 'decoder': 'final',
                  'decoder_sha256': FINAL_DECODER_SHA, 'resolution': 256, 'zero_level': 0, 'spacing': 2 / 255,
                  'component_filtering': 'none', 'grid_frame': 'decoder normalized [-1,1]^3', 'source_array_row': row,
                  'finite_input_code': bool(np.isfinite(codes[row]).all())}
        t0 = time.time()
        try:
            if not record['finite_input_code']:
                raise ValueError('nonfinite latent code')
            field = sdf_field(codes[row])
            record.update(sdf_min=float(field.min()), sdf_max=float(field.max()))
            vertices, faces, extractor = bench.extract(field, 'skimage')
            record.update(bench.mesh_info(vertices, faces))
            np.savez(folder / 'mesh.npz', vertices=vertices, faces=faces)   # temporary file
            record.update(status='completed', extractor=extractor, mesh_path=f'tasks/{task}/mesh.npz',
                          output_sha256={'mesh.npz': sha(folder / 'mesh.npz')})
        except Exception as error:      # every failure stays in the denominator
            record.update(status='mesh_failed', failure_type=type(error).__name__, failure=str(error))
        record['task_wall_seconds'] = time.time() - t0
        records.append(record)
        if done % 20 == 0 or done == len(rows):
            rate = (time.time() - start) / done
            log(f'  decoded {done}/{len(rows)} meshes ({rate:.1f} s/mesh)')
    write_json(root / 'summary.json', {'schema': 'fm_final_ablation_decode_v1', 'records': records,
                                       'resolution': 256, 'valid_meshes': sum(r['status'] == 'completed' for r in records)})
    write_json(root / 'run_config.json', {'tasks': [{'task': r['task']} for r in records], 'paired_reference': 'final',
                                          'training_seed': seed, 'arm': arm, 'method': method,
                                          'codes_sha256': sha(codes_path), 'decoder_parity_max_abs': decoder()[2]})
    return root


# ---------------------------------------------------------------- frozen metrics -------------
_vendor = None


def vendor():
    """Frozen vendor_metrics.py, unchanged bytes. Like the six-arm adapter, the accepted method-name
    registry is extended. Two speed-ups leave every value unchanged: reference loading is cached, and the exact
    KD-tree queries use all CPU cores (validation check 1 confirms the six-arm study's per-skull values)."""
    global _vendor
    if _vendor is None:
        vm = load_module(MESHSRC / 'vendor' / 'vendor_metrics.py', 'fm_vendor_metrics')
        vm.METHODS = tuple(vm.METHODS) + tuple(a['method'] for a in ARMS.values())
        original, cache = vm.load_references, {}

        def load_references(manifest):
            key = str(Path(manifest).resolve())
            if key not in cache:
                cache[key] = original(manifest)
            return cache[key]

        from scipy.spatial import cKDTree

        def cd(left, right, left_tree=None, right_tree=None):
            """The frozen evaluator's exact KD-tree Chamfer distance; only the query uses all CPU cores."""
            left_tree = cKDTree(left) if left_tree is None else left_tree
            right_tree = cKDTree(right) if right_tree is None else right_tree
            value = float(np.square(left_tree.query(right, workers=-1)[0]).mean()
                          + np.square(right_tree.query(left, workers=-1)[0]).mean())
            if not np.isfinite(value) or value < 0:
                raise ValueError('Invalid Chamfer value')
            return value

        vm.load_references, vm.cd = load_references, cd
        _vendor = vm
    return _vendor


def evaluate_dir(decode_root, out):
    out = Path(out)
    if (out / 'summary.json').exists():
        return json.loads((out / 'summary.json').read_text())
    if out.exists():
        shutil.rmtree(out)
    start = time.time()
    result = vendor().run(REFS / 'reference_manifest.json', [decode_root], out, points=10000,
                          surface_seeds=SURFACE_SEEDS, resume=False)
    shutil.rmtree(out / 'surface_samples', ignore_errors=True)     # large, regenerable
    log(f'  metrics done in {time.time() - start:.0f} s')
    return result


def evaluate(seed, arm, decode_root):
    return evaluate_dir(decode_root, OUT / 'metrics' / f'seed{seed}_{arm}')


def brief(result, arm, seed):
    rows = []
    for g in sorted(result['groups'], key=lambda g: (g['cohort'] != 'unconditional_species_mixture', g['cohort'])):
        s = g.get('geometry_surface_seed_summary', {})
        f = g.get('morphfid_decoder_frame_primary', {})
        nn = s.get('generated_to_reference_nn_CD_mean', {}).get('mean', float('nan'))
        mmd = s.get('MMD_CD_reference_to_generated', {}).get('mean', float('nan'))
        cov = s.get('COV_reference_fraction', {}).get('mean', float('nan'))
        div = s.get('diversity_all_pairs_CD_mean', {}).get('mean', float('nan'))
        f8 = f.get('geometry8_no_count_or_centroid', {}).get('value', float('nan')) if isinstance(f, dict) else float('nan')
        cohort = g['cohort'].replace('unconditional_species_mixture', 'mixture').replace('conditional_', '')
        ref_div = REAL_WITHIN.get(cohort, REAL_MIXTURE_DIVERSITY)
        rows.append(f"    {cohort:15s} valid {g['valid_meshes']:3d}/{g['attempts']:3d}  nn-CD {nn * 1e3:5.2f}  "
                    f"MMD {mmd * 1e3:5.2f}  COV {cov:4.2f}  div {div * 1e3:5.2f} ({100 * div / ref_div:5.1f}% of real)  "
                    f"MorphFID8 {f8:5.2f}")
    log(f'{arm} seed {seed} (x1e-3; real skulls: nn-CD {REAL_LOO_NN * 1e3:.2f}, mixture diversity '
        f'{REAL_MIXTURE_DIVERSITY * 1e3:.2f})\n' + '\n'.join(rows))


# ---------------------------------------------------------------- validation -----------------
# Frozen per-skull nearest-reference CDs (surface seed 20261014) of the six-arm study's
# no-augmentation arm, mixture slots 0-3 (final decoder).
FROZEN_NO_AUG_NN = [0.0002967216158403156, 0.000409004300481227, 0.0005300674325193753, 0.00035709577309446365]


def validate():
    """Three checks before the long run (about 10 minutes):
    1. decoding + frozen metrics reproduce the six-arm study's per-skull scores for four of its skulls;
    2. our sampler equals the original sampler on the retrained no-augmentation arm;
    3. retraining that arm reproduces the six-arm study's codes (informational across GPU types)."""
    restore()
    vdir = OUT / 'validation'
    done = vdir / 'VALIDATION.json'
    if done.exists():
        old = json.loads(done.read_text())
        if (old.get('decode_and_metric_reproduction', {}).get('max_relative_difference', 1) <= 1e-3
                and old.get('our_sampler_vs_original_sampler_max_abs', 1) <= 1e-5):
            log('checks already passed in an earlier session (restored); skipping them')
            return old
    report = {}
    vdir.mkdir(parents=True, exist_ok=True)
    frozen_path = vdir / 'six_arm_no_augmentation_codes.npz'
    with zipfile.ZipFile(DL / 'full10000_no_augmentation_generation140.zip') as z:
        frozen_path.write_bytes(z.read('full10000_samples/no_augmentation/GENERATED_CODES.npz'))
    frozen = np.load(frozen_path, allow_pickle=False)['codes']
    smoke_root, smoke_out = vdir / 'smoke_decode', vdir / 'smoke_metrics'
    if not (smoke_out / 'summary.json').exists():
        decode_arm(42, 'species_residual', frozen_path, smoke_root, rows=[0, 1, 40, 41, 42, 43])
    result = evaluate_dir(smoke_root, smoke_out)
    shutil.rmtree(smoke_root / 'tasks', ignore_errors=True)
    mixture = [g for g in result['groups'] if g['cohort'] == 'unconditional_species_mixture'][0]
    obtained = mixture['geometry_by_surface_seed'][0]['nearest_reference_CD_each_generated']
    checks = [{'slot': k, 'expected': e, 'obtained': o, 'relative_difference': abs(o - e) / e}
              for k, (e, o) in enumerate(zip(FROZEN_NO_AUG_NN, obtained))]
    worst = max(c['relative_difference'] for c in checks)
    report['decode_and_metric_reproduction'] = {'per_skull': checks, 'max_relative_difference': worst,
                                                'smoke_groups_evaluated': len(result['groups'])}
    log(f'check 1 (decode + frozen metrics vs six-arm study, 4 skulls): max relative difference {worst:.2e}')
    if worst > 1e-3:
        write_json(vdir / 'VALIDATION.json', report)
        raise RuntimeError('Decode/metric reproduction failed (difference > 0.1%). Stop and send VALIDATION.json to Claude.')
    train(42, 'no_augmentation')
    official_dir = vdir / 'original_sampler_no_augmentation_seed42'
    if not (official_dir / 'GENERATED_CODES.npz').exists():
        if official_dir.exists():
            shutil.rmtree(official_dir)
        rmc().sample(str(prepared_path(42)), str(trained_path(42, 'no_augmentation')), str(official_dir))
    official = np.load(official_dir / 'GENERATED_CODES.npz')['codes']
    mine = np.load(sample(42, 'species_residual', vdir / 'our_sampler_guidance1.5', guidance=1.5))['codes']
    report['our_sampler_vs_original_sampler_max_abs'] = float(np.abs(mine - official).max())
    report['retrained_vs_six_arm_study_codes_max_abs'] = float(np.abs(official - frozen).max())
    report['six_arm_codes_std'] = float(frozen.std())
    log(f'check 2 (our sampler vs original sampler): max |diff| {report["our_sampler_vs_original_sampler_max_abs"]:.2e}')
    log(f'check 3 (retrained no-augmentation arm vs six-arm study): max |diff| '
        f'{report["retrained_vs_six_arm_study_codes_max_abs"]:.2e} (code std {report["six_arm_codes_std"]:.3f}; '
        'small differences are normal on a different GPU type)')
    write_json(vdir / 'VALIDATION.json', report)
    if report['our_sampler_vs_original_sampler_max_abs'] > 1e-5:
        raise RuntimeError('Our sampler differs from the original sampler. Stop and send VALIDATION.json to Claude.')
    log('all checks passed')
    return report


# ---------------------------------------------------------------- main loop ------------------
def package(tag):
    path = ROOT / f'PhyloSDF_final_ablation_{tag}.zip'
    tmp = path.with_name(path.name + '.part')
    with zipfile.ZipFile(tmp, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob('*')):
            text = str(p)
            if p.is_file() and '/tasks/' not in text and '/surface_samples/' not in text and not p.name.endswith('.tmp'):
                z.write(p, Path('results') / p.relative_to(OUT))
        for p in sorted((WORK / 'final').rglob('*')):
            if p.is_file() and (p.suffix == '.json' or p.name in ('DATA.npz', 'GENERATION_INPUTS.npz')):
                z.write(p, Path('work') / p.relative_to(WORK))
    tmp.replace(path)            # never leaves a half-written zip behind
    return path


def keep_figure_meshes(seed, arm, root, per_species=3):
    if seed != 42 or arm != 'pooled_residual':
        return
    target = OUT / 'figure_meshes' / f'seed{seed}_{arm}'
    for task in sorted((Path(root) / 'tasks').iterdir()):
        mesh = task / 'mesh.npz'
        if mesh.exists() and any(f'conditional_{sp}__slot{k:03d}' in task.name for sp in FOCAL for k in range(per_species)):
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy(mesh, target / (task.name + '.npz'))


# Seed 42 of every arm first, so even a stopped run leaves one complete ablation table; then seeds 43 and 44.
ORDER = ([(42, arm) for arm in ('pooled_residual', 'species_residual', 'species_fullcode',
                                'pooled_residual_no_var', 'pooled_residual_aug', 'pooled_residual_tcal')]
         + [(seed, arm) for seed in (43, 44) for arm in ('pooled_residual', 'species_residual', 'species_fullcode')]
         + [(seed, arm) for seed in (43, 44) for arm in ('pooled_residual_no_var', 'pooled_residual_aug',
                                                       'pooled_residual_tcal')])


def completed(seed, arm):
    return (OUT / 'metrics' / f'seed{seed}_{arm}' / 'summary.json').exists()


UPLOAD_DIR = Path('/content')


def restore():
    """Continue from a progress zip uploaded to /content (results only; training caches are rebuilt, deterministically)."""
    zips = sorted(UPLOAD_DIR.glob('PhyloSDF_final_ablation*.zip'), key=lambda p: p.stat().st_size)
    if not zips:
        return 0
    path, n = zips[-1], 0
    with zipfile.ZipFile(path) as z:
        for member in z.namelist():
            parts = Path(member).parts
            if not member.startswith('results/') or member.endswith('/') or '..' in parts:
                continue
            target = OUT / Path(*parts[1:])
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(z.read(member))
                n += 1
    done = [f'{arm} seed {seed}' for seed, arm in ORDER if completed(seed, arm)]
    log(f'restored {n} files from {path.name}; already complete: ' + (', '.join(done) or 'none'))
    return n


def run_arm(seed, arm):
    metrics_summary = OUT / 'metrics' / f'seed{seed}_{arm}' / 'summary.json'
    if metrics_summary.exists():
        brief(json.loads(metrics_summary.read_text()), arm, seed)
        return
    start = time.time()
    log(f'=== seed {seed} | {arm} ===')
    train(seed, ARMS[arm]['variant'])
    codes_dir = OUT / 'codes' / f'seed{seed}' / arm
    codes = codes_dir / 'GENERATED_CODES.npz'
    if not codes.exists():
        sample(seed, arm, codes_dir)
    root = OUT / 'decode' / f'seed{seed}_{arm}'
    decode_arm(seed, arm, codes, root)
    result = evaluate(seed, arm, root)
    keep_figure_meshes(seed, arm, root)
    shutil.rmtree(root / 'tasks', ignore_errors=True)
    brief(result, arm, seed)
    log(f'{arm} seed {seed} finished in {(time.time() - start) / 60:.1f} min')


def run(order=ORDER, download=True):
    """Runs every (seed, arm) in order; after each one a progress zip is saved in /content/FM (download it from the
    Files panel at any time). Completed runs, including ones restored from an uploaded zip, are skipped."""
    restore()
    t0 = time.time()
    for k, (seed, arm) in enumerate(order, 1):
        if not completed(seed, arm):
            prepared_path(seed)
        run_arm(seed, arm)
        path = package('progress')
        log(f'[{k}/{len(order)}] saved {path} ({path.stat().st_size / 2 ** 20:.1f} MB) - download it from the Files '
            f'panel (folder icon > FM) whenever you like; elapsed {(time.time() - t0) / 3600:.1f} h')
    path = package('ALL')
    log(f'all runs done; saved {path} ({path.stat().st_size / 2 ** 20:.1f} MB)')
    if download:
        try:
            from google.colab import files
            files.download(str(path))
        except Exception as error:
            log(f'automatic download failed ({error}); use the Files panel to download {path}')
