# fmfix_engine.py -- Table 4b's species-conditioned rows rerun with CORRECTED size inputs.
#
# The submitted pipeline read each skull's size (bounding-box length, width, height) from size.xlsx, whose Filename
# column had been re-sorted on its own; 88 of the 100 skulls received another skull's size. The final model (pooled
# residual flow) never uses size, but the species-conditioned arms of Table 4b do (species token + size + phylogeny,
# as in the submitted model). This engine is fm_engine.py (v2) with one change: the size of every skull is replaced by
# the bounding-box extents of its own mesh (identical to Dataset.xlsx of github.com/kaikwanlau/skull-morphology),
# checked against the downloaded reference meshes. Everything else (protocol, seeds, initial weights, noise, decoding,
# frozen metrics) is unchanged. Arms:
#   species_residual_fixsize    residual flow conditioned on species + size + phylogeny (Table 4b '+ species input')
#   species_fullcode_fixsize    direct generation on full codes, same conditioning (Table 4b 'standard CFM')
#   species_residual_tokenonly  residual flow conditioned on the species token only (no size, no phylogeny)
#
# Original description of fm_engine.py (v2) follows.
#
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

ROOT = Path('/content/FMFIX')
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
    'fm_species_residual_tokenonly': {'augmentation': False, 'phylo_conditioning': False, 'target': 'residual',
                                      'var_weight': .1, 'zero_size': True},
}
ARMS = {
    'species_residual_fixsize': dict(variant='no_augmentation', neutral=False, guidance=1.0, temperature=1.0,
                                     method='fm_species_residual_fixsize'),
    'species_fullcode_fixsize': dict(variant='fm_species_fullcode', neutral=False, guidance=1.0, temperature=1.0,
                                     method='fm_species_fullcode_fixsize'),
    'species_residual_tokenonly': dict(variant='fm_species_residual_tokenonly', neutral=False, guidance=1.0,
                                       temperature=1.0, method='fm_species_residual_tokenonly'),
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
STUDY_GROUP, STUDY_ARM = 'final_model_ablation_fixsize_v1', 'matched_fixed_protocol'
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
    write_json(OUT / 'SIZE_CORRECTION.json', check_sizes())
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
        original_legacy = module.load_legacy

        def load_legacy(base, pair, device):
            """The verified original inputs, with every skull's size replaced by its own mesh's extents."""
            latent, species, bbox, phylo, matrix, metadata, decoder, manifest_sha = original_legacy(base, pair, device)
            fixed = corrected_sizes(bbox.dtype)
            if fixed.shape != bbox.shape:
                raise RuntimeError('corrected size array has the wrong shape')
            return latent, species, fixed, phylo, matrix, metadata, decoder, manifest_sha

        def training_tensors(arrays, variant):
            tensors, weights = original(arrays, variant)
            if module.VARIANTS[variant].get('neutral'):
                tensors['species'] = torch.full_like(tensors['species'], 24)
                tensors['bbox'] = torch.zeros_like(tensors['bbox'])
                tensors['phylo'] = torch.zeros_like(tensors['phylo'])
            if module.VARIANTS[variant].get('zero_size'):
                tensors['bbox'] = torch.zeros_like(tensors['bbox'])
                tensors['phylo'] = torch.zeros_like(tensors['phylo'])
            return tensors, weights

        module.load_legacy = load_legacy
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
        if cfg.get('zero_size'):
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


def corrected_sizes(dtype=np.float32):
    names = species_specimen_names()
    if names != CORRECT_SIZE_NAMES:
        raise RuntimeError('specimen order differs from the embedded corrected sizes')
    return np.asarray(CORRECT_SIZE_MM, dtype=dtype)


def species_specimen_names():
    meta = json.loads((SRC / 'inputs' / 'metadata.json').read_text())
    return [s['filename'] for s in meta['specimens']]


def _read_stl_vertices(path):
    data = Path(path).read_bytes()
    n = int(np.frombuffer(data[80:84], '<u4')[0])
    rec = np.frombuffer(data[84:84 + 50 * n], dtype=np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')]))
    return rec['v'].reshape(-1, 3).astype(np.float64)


def check_sizes():
    """Corrected sizes = bounding-box extents of each downloaded reference mesh (the PhyloSDF training meshes)."""
    rows = {r['specimen']: r for r in json.loads((REFS / 'reference_manifest.json').read_text())['references']}
    fixed = corrected_sizes(np.float64)
    old = np.load(SRC / 'inputs' / 'bbox_dims.npy').astype(np.float64)
    worst, changed = 0.0, 0
    for i, name in enumerate(CORRECT_SIZE_NAMES):
        v = _read_stl_vertices(REFS / rows[name]['raw_stl'])
        ext = v.max(0) - v.min(0)
        worst = max(worst, float(np.abs(ext - fixed[i]).max()))
        changed += bool(np.abs(old[i] - fixed[i]).max() > 1e-3)
    if worst > 1e-3:
        raise RuntimeError(f'corrected sizes differ from the reference meshes by {worst:.2e} mm')
    log(f'size check: corrected sizes equal the mesh extents (max |diff| {worst:.1e} mm); {changed}/100 skulls change')
    return {'max_abs_difference_to_mesh_extents_mm': worst, 'skulls_whose_size_changes': changed,
            'source': 'axis-aligned bounding box of each reference STL (= Dataset.xlsx length_x, width_y, height_z)'}


def v2_pooled_codes():
    import base64, io
    raw = base64.b64decode(''.join(V2_POOLED_S42_B64.split()))
    if hashlib.sha256(raw).hexdigest() != V2_POOLED_S42_SHA:
        raise RuntimeError('embedded reference codes changed')
    return np.load(io.BytesIO(raw))['codes']


def validate():
    """Checks before the long run (about 15 minutes):
    1. decoding + frozen metrics reproduce the six-arm study's per-skull scores for four of its skulls;
    2. the corrected sizes equal each training mesh's extents;
    3. the final model (which never sees size) is unchanged: retrained with the corrected-size preparation, seed 42
       reproduces the codes of the final-model ablation (informational across GPU types);
    4. our sampler equals the original sampler on the species-conditioned arm."""
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
        decode_arm(42, 'species_residual_fixsize', frozen_path, smoke_root, rows=[0, 1, 40, 41, 42, 43])
    result = evaluate_dir(smoke_root, smoke_out)
    shutil.rmtree(smoke_root / 'tasks', ignore_errors=True)
    mixture = [g for g in result['groups'] if g['cohort'] == 'unconditional_species_mixture'][0]
    obtained = mixture['geometry_by_surface_seed'][0]['nearest_reference_CD_each_generated']
    checks = [{'slot': k, 'expected': e, 'obtained': o, 'relative_difference': abs(o - e) / e}
              for k, (e, o) in enumerate(zip(FROZEN_NO_AUG_NN, obtained))]
    worst = max(c['relative_difference'] for c in checks)
    report['decode_and_metric_reproduction'] = {'per_skull': checks, 'max_relative_difference': worst}
    log(f'check 1 (decode + frozen metrics vs six-arm study, 4 skulls): max relative difference {worst:.2e}')
    if worst > 1e-3:
        write_json(vdir / 'VALIDATION.json', report)
        raise RuntimeError('Decode/metric reproduction failed (difference > 0.1%). Stop and send VALIDATION.json to Claude.')
    report['sizes'] = check_sizes()
    train(42, 'fm_pooled_residual')
    pooled = np.load(sample(42, 'pooled_residual', vdir / 'pooled_residual_seed42'))['codes']
    report['final_model_codes_vs_final_ablation_max_abs'] = float(np.abs(pooled - v2_pooled_codes()).max())
    log(f'check 3 (final model unaffected by the size correction): max |code diff| vs the final-model ablation '
        f'{report["final_model_codes_vs_final_ablation_max_abs"]:.2e} (code std {float(pooled.std()):.3f}; tiny '
        'differences are normal on a different GPU type)')
    train(42, 'no_augmentation')
    official_dir = vdir / 'original_sampler_no_augmentation_seed42'
    if not (official_dir / 'GENERATED_CODES.npz').exists():
        if official_dir.exists():
            shutil.rmtree(official_dir)
        rmc().sample(str(prepared_path(42)), str(trained_path(42, 'no_augmentation')), str(official_dir))
    official = np.load(official_dir / 'GENERATED_CODES.npz')['codes']
    mine = np.load(sample(42, 'species_residual_fixsize', vdir / 'our_sampler_guidance1.5', guidance=1.5))['codes']
    report['our_sampler_vs_original_sampler_max_abs'] = float(np.abs(mine - official).max())
    report['corrected_vs_original_size_codes_max_abs'] = float(np.abs(official - frozen).max())
    log(f'check 4 (our sampler vs original sampler): max |diff| {report["our_sampler_vs_original_sampler_max_abs"]:.2e}; '
        f'codes with corrected vs original sizes differ by up to {report["corrected_vs_original_size_codes_max_abs"]:.3f} '
        '(expected: the size input changed)')
    write_json(vdir / 'VALIDATION.json', report)
    if report['our_sampler_vs_original_sampler_max_abs'] > 1e-5:
        raise RuntimeError('Our sampler differs from the original sampler. Stop and send VALIDATION.json to Claude.')
    log('all checks passed')
    return report


# ---------------------------------------------------------------- main loop ------------------
def package(tag):
    path = ROOT / f'PhyloSDF_fixsize_{tag}.zip'
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
ORDER = [(seed, arm) for seed in (42, 43, 44)
         for arm in ('species_residual_fixsize', 'species_fullcode_fixsize', 'species_residual_tokenonly')]


def completed(seed, arm):
    return (OUT / 'metrics' / f'seed{seed}_{arm}' / 'summary.json').exists()


UPLOAD_DIR = Path('/content')


def restore():
    """Continue from a progress zip uploaded to /content (results only; training caches are rebuilt, deterministically)."""
    zips = sorted(UPLOAD_DIR.glob('PhyloSDF_fixsize*.zip'), key=lambda p: p.stat().st_size)
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
            f'panel (folder icon > FMFIX) whenever you like; elapsed {(time.time() - t0) / 3600:.1f} h')
    path = package('ALL')
    log(f'all runs done; saved {path} ({path.stat().st_size / 2 ** 20:.1f} MB)')
    if download:
        try:
            from google.colab import files
            files.download(str(path))
        except Exception as error:
            log(f'automatic download failed ({error}); use the Files panel to download {path}')

# ---------------------------------------------------------------- embedded data --------------
# Corrected size of every skull (mm; length_x, width_y, height_z): the axis-aligned bounding box of its own mesh, in the
# training order of metadata.json. Equal to Dataset.xlsx of github.com/kaikwanlau/skull-morphology.
CORRECT_SIZE_NAMES = ["C.PallidusA", "C.PallidusB", "C.PallidusC", "C.PallidusD", "C.ParvulusA", "C.ParvulusB", "C.ParvulusC", "C.ParvulusD", "C.ParvulusE", "C.PsittaculaA", "C.PsittaculaB", "C.PsittaculaC", "C.PsittaculaD", "C.flaveolaA", "C.flaveolaB", "C.flaveolaC", "C.flaveolaD", "C.flaveolaE", "C2.FuscaA", "C2.FuscaB", "C2.OlivaceaA", "C2.OlivaceaB", "C2.OlivaceaC", "E.campestrisA", "E.campestrisB", "E.campestrisC", "E.campestrisD", "E.campestrisE", "G.ConirostrisA", "G.ConirostrisB", "G.ConirostrisC", "G.ConirostrisD", "G.ConirostrisE", "G.ConirostrisF", "G.ConirostrisG", "G.DifficilisA", "G.DifficilisB", "G.FortisA", "G.FortisB", "G.FortisC", "G.FortisD", "G.FortisE", "G.FuliginosaA", "G.FuliginosaB", "G.FuliginosaC", "G.FuliginosaD", "G.FuliginosaE", "G.FuliginosaF", "G.MagnirostrisA", "G.MagnirostrisB", "G.MagnirostrisC", "G.MagnirostrisD", "G.MagnirostrisE", "G.ScandensA", "G.ScandensB", "G.SeptentrionalistA", "G.SeptentrionalistB", "L.anoxanthusA", "L.anoxanthusB", "L.anoxanthusC", "L.anoxanthusD", "L.anoxanthusE", "L.noctisA", "L.noctisB", "L.noctisC", "L.noctisE", "L.portoricensisA", "L.portoricensisB", "L.portoricensisC", "L.portoricensisD", "L.portoricensisE", "L.violaceaA", "L.violaceaB", "L.violaceaC", "L.violaceaD", "L.violaceaE", "M.nigraA", "M.nigraC", "M.nigraD", "M.nigraE", "P.CrassirostrisB", "P.CrassirostrisC", "P.CrassirostrisD", "P.CrassirostrisE", "P2.InornataA", "P2.InornataB", "T.BicolorA", "T.BicolorB", "T.BicolorC", "T.BicolorD", "T.BicolorE", "T.canoraA", "T.canoraB", "T.canoraC", "T.canoraD", "T.canoraE", "T.olivaceaB", "T.olivaceaC", "T.olivaceaD", "T.olivaceaE"]
CORRECT_SIZE_MM = [[33.215496, 17.382399, 16.795805], [32.929001, 16.971269, 15.724266], [33.028788, 17.054966, 15.780712], [33.028828, 17.054987, 15.780713], [25.448855, 15.804236, 15.550212], [26.496113, 15.33464, 15.158322], [24.95489, 14.926009, 13.87183], [24.954994, 14.926071, 13.871888], [24.482099, 15.753777, 15.648098], [29.408774, 18.660337, 17.912613], [28.575351, 17.407166, 16.443836], [28.381917, 16.26942, 15.25787], [28.123603, 16.116239, 15.394715], [27.784896, 13.07499, 11.911357], [26.482409, 12.738422, 11.572667], [26.630806, 12.631825, 12.474127], [28.237201, 13.355453, 11.80992], [28.417264, 13.429103, 11.798685], [24.78668, 13.976942, 12.62544], [25.146867, 13.382392, 12.058603], [26.861537, 14.279736, 12.507781], [25.989511, 14.086778, 11.688623], [27.049004, 13.985115, 12.094317], [31.279086, 14.57874, 13.53045], [31.166288, 14.920727, 13.765692], [30.553099, 14.30683, 13.884888], [29.556677, 14.223228, 13.223026], [29.292737, 14.054849, 12.703552], [33.96495, 18.267929, 16.693438], [35.834797, 17.517889, 15.815438], [35.171981, 20.047633, 18.742229], [35.667445, 19.637973, 18.321999], [35.965683, 19.966125, 18.719872], [32.587329, 18.108942, 18.349255], [35.650325, 20.152917, 19.172789], [27.641973, 15.433545, 15.209097], [27.310783, 15.331369, 16.678737], [30.911628, 16.943285, 16.627182], [31.299546, 17.624324, 16.615941], [33.174639, 19.659266, 18.98068], [31.535275, 18.094078, 18.975431], [29.586675, 16.166284, 15.89713], [26.860279, 15.458116, 14.171734], [26.691727, 15.473843, 14.136262], [24.85549, 15.685118, 16.036819], [26.429254, 15.347324, 14.396963], [26.734119, 14.838145, 14.215234], [25.540702, 15.480179, 16.181153], [37.776474, 24.031554, 20.255352], [36.927464, 23.516075, 19.866896], [36.003769, 24.26674, 21.714206], [37.069497, 23.851021, 23.007429], [37.498923, 23.108389, 21.055562], [34.898864, 17.156633, 14.983153], [34.871084, 17.74591, 15.789656], [30.262746, 16.390247, 15.29416], [30.665527, 16.150158, 14.770782], [24.171363, 13.491234, 13.439883], [23.606173, 13.551296, 12.812455], [23.600783, 13.583867, 12.764111], [24.09047, 13.575391, 13.431773], [24.060303, 13.490038, 12.453138], [28.69867, 15.080952, 15.042797], [28.59376, 15.20371, 15.19633], [28.627799, 14.74457, 14.576816], [27.723201, 15.141213, 14.897704], [32.658446, 18.772185, 18.490719], [33.96281, 18.895631, 19.260614], [32.950842, 18.953332, 18.706031], [31.792066, 19.415947, 18.332383], [33.411001, 17.987135, 17.944918], [31.857317, 17.963713, 17.213292], [31.723192, 18.459029, 17.825169], [31.339976, 17.487989, 16.994226], [33.587442, 18.383274, 18.143749], [32.112296, 17.762749, 17.227894], [27.323431, 15.096241, 15.316404], [27.125063, 14.705216, 15.15797], [27.216508, 14.392592, 15.006132], [26.51557, 14.666758, 14.488958], [30.47705, 18.167127, 17.234424], [31.681653, 18.081208, 17.073681], [31.401782, 18.727551, 16.775068], [32.63192, 19.144978, 17.217699], [27.569541, 13.760966, 14.320283], [27.554919, 13.741684, 13.893467], [21.649636, 14.159708, 13.272029], [21.798077, 13.725001, 13.779637], [21.611903, 13.475446, 14.035443], [22.152703, 13.790419, 13.052871], [23.079736, 13.646571, 12.715502], [20.215377, 12.388683, 11.636874], [21.341728, 13.218876, 11.632504], [20.775399, 12.585073, 11.347019], [20.371298, 12.229728, 11.459067], [21.859589, 12.499156, 11.026688], [21.952329, 13.117521, 11.464268], [21.345675, 12.395136, 12.091194], [22.89023, 13.085347, 12.328676], [22.349862, 12.862599, 12.016947]]

# Generated codes of the final model, seed 42, from the final-model ablation (for validation check 3).
V2_POOLED_S42_SHA = 'c8947d92063b440bcd017e658a02e521a40c89618904165b451b2948bbb49063'
V2_POOLED_S42_B64 = """
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
"""
