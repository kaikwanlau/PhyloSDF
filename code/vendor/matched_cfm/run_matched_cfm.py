"""New fixed-update, matched CFM component study using frozen original assets.

This is an additive experiment, not a continuation of historical training.
It does not change original Drive files, runtimes, models or checkpoints.
"""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse, hashlib, importlib.util, json, random, sys, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from resume_state import capture_rng, restore_rng, isolated_rng, atomic_torch_save, verified_torch_load

VARIANTS = {
    'full': {'augmentation': True, 'phylo_conditioning': True, 'target': 'residual', 'var_weight': .1},
    'no_augmentation': {'augmentation': False, 'phylo_conditioning': True, 'target': 'residual', 'var_weight': .1},
    'no_phylo_conditioning': {'augmentation': True, 'phylo_conditioning': False, 'target': 'residual', 'var_weight': .1},
    'no_variance_penalty': {'augmentation': True, 'phylo_conditioning': True, 'target': 'residual', 'var_weight': 0.},
    'full_code_with_variance': {'augmentation': True, 'phylo_conditioning': True, 'target': 'full_code', 'var_weight': .1},
    'standard_cfm': {'augmentation': True, 'phylo_conditioning': True, 'target': 'full_code', 'var_weight': 0.},
}
PROTOCOL = {
    'schema': 'matched_cfm_fixed_updates_protocol_v1',
    'architecture': {'latent_dim': 64, 'model_dim': 128, 'num_layers': 3, 'num_species': 24,
                     'condition_on_species': True, 'condition_on_size': True, 'condition_on_phylo': True,
                     'species_embed_dim': 32, 'dropout': .05},
    'training_seeds': [42, 43, 44], 'pilot_updates': 100, 'full_updates': 10000,
    'batch_size': 256, 'original_item_weight': 3., 'augmented_item_weight': 1.,
    'optimizer': {'name': 'AdamW', 'lr': .0001, 'weight_decay': .00001},
    'schedule': {'name': 'CosineAnnealingLR_per_update', 'eta_min': .000001},
    'gradient_clip_norm': 1., 'whole_batch_species_dropout_probability': .1,
    'augmentation': {'num_augmented_per_specimen': 10, 'arap_noise_std': .1,
                     'interpolation_samples': 5, 'total_target': 2000},
    'augmentation_route': 'original decoder finite-difference Hessian, phylo interpolation and centroid extrapolation; no PCA fallback',
    'matrix': 'saved explicitly labelled historical taxonomy proxy unchanged; not a molecular phylogeny',
    'centroids': '100 real checkpoint-native codes only, same centroids for training and generation',
    'normalization': '100 real codes only; vector mean and scalar unbiased global std for each target; real-only bbox vector mean/std',
    'variance_penalty': 'relu(target.var()-midpoint_one_step_estimate.var()), pooled over batch and coordinates',
    'zero_penalty_control': 'midpoint forward still evaluated to preserve dropout RNG consumption',
    'conditioning_ablation': 'zero phylo embedding only; preserve architecture/initial weights; DeepSDF and augmentation still contain proxy information',
    'generation': {'Euler_steps': 50, 'species_guidance_scale': 1.5, 'temperature': 1.,
                   'size': 'real mean bbox within the requested species, normalized with real-only saved stats',
                   'conditional_species': ['G.Fortis', 'G.Magnirostris', 'G.Fuliginosa', 'C.Parvulus'],
                   'samples_per_conditional_species': 10, 'mixture_samples': 100,
                   'mixture_schedule': 'explicit species index slot modulo24; includes between-species variation'},
    'changes_from_historical_trainer': [
        '10,000 optimizer updates replaces 10,000 epochs; all arms use batch256 with replacement',
        'real centroids replace augmented-dataset centroids in training, resolving historical train/generate mismatch',
        'normalization fitted on real codes only and frozen across augmentation toggle',
        'correct checkpoint-native best/final latent library and matching augmentation decoder are required',
        'explicit generation size inputs and temperature1 replace historical absent bbox/default temperature behavior',
        'predeclared final fixed-budget checkpoints replace training-loss best checkpoint selection',
        'augmentation dataset and paired initialization are frozen separately from training RNG',
    ],
    'claims': 'pilot is execution/resume validation only; comparisons describe this new matched protocol; no performance claim from pilot',
}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.partial')
    with temp.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    temp.replace(path)


def source_closure():
    return {str(p.relative_to(HERE)): file_sha(p) for p in
            [HERE / 'run_matched_cfm.py', HERE / 'resume_state.py',
             HERE / 'vendor/diffusion.py', HERE / 'vendor/augmentation.py', HERE / 'vendor/deepsdf.py']}


def load_vendor(name):
    path = HERE / 'vendor' / (name + '.py')
    spec = importlib.util.spec_from_file_location('matched_frozen_' + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def deterministic():
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False


def state_sha(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode()); digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode()); digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def runtime_identity(device):
    return {'torch': torch.__version__, 'numpy': np.__version__, 'threads': torch.get_num_threads(),
            'cuda': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(torch.device(device)) if str(device).startswith('cuda') else None}


def verify_inputs(base=HERE):
    base = Path(base)
    manifest_path = base / 'INPUT_MANIFEST.json'
    manifest = json.loads(manifest_path.read_text())
    need(manifest['schema'] == 'matched_cfm_grounded_inputs_v1', 'Unexpected input inventory')
    for row in manifest['files']:
        p = base / row['path']
        need(p.is_file() and not p.is_symlink() and p.stat().st_size == row['size_bytes'] and
             file_sha(p) == row['sha256'], 'Frozen input differs: ' + row['path'])
    return manifest, file_sha(manifest_path)


def load_legacy(base, pair, device):
    manifest, manifest_sha = verify_inputs(base)
    root = Path(base) / 'inputs'
    need(pair in ('best', 'final'), 'Decoder pair must be best or final')
    checkpoint = torch.load(root / (pair + '.pt'), map_location='cpu', weights_only=False)
    need('latent_lib_state_dict' in checkpoint and 'decoder_state_dict' in checkpoint,
         'Checkpoint-native decoder and latent library required')
    latent = checkpoint['latent_lib_state_dict']['latent_codes.weight'].detach().cpu().numpy()
    species = np.load(root / 'species_indices.npy', allow_pickle=False)
    bbox = np.load(root / 'bbox_dims.npy', allow_pickle=False)
    phylo = np.load(root / 'phylo_embeddings.npy', allow_pickle=False)
    matrix = np.load(root / 'phylo_dist.npy', allow_pickle=False)
    metadata = json.loads((root / 'metadata.json').read_text())
    need(latent.shape == (100, 64) and species.shape == (100,) and bbox.shape == (100, 3),
         'Frozen100 specimen arrays differ')
    need(phylo.ndim == 2 and phylo.shape[0] == 24 and matrix.shape == (24, 24),
         'Frozen24 species conditioning differs')
    need(np.isfinite(latent).all() and np.isfinite(bbox).all() and np.isfinite(phylo).all(), 'Nonfinite input')
    need(np.array_equal(species, [s['species_idx'] for s in metadata['specimens']]),
         'Metadata/specimen indices disagree')
    need(np.array_equal(bbox, np.asarray([s['bbox_dims'] for s in metadata['specimens']], dtype=bbox.dtype)),
         'Metadata/specimen bbox disagrees')
    labels = [metadata['idx_to_species'][str(i)] for i in range(24)]
    need(all(metadata['species_to_idx'][s] == i for i, s in enumerate(labels)), 'Explicit species order differs')
    decoder_class = load_vendor('deepsdf').DeepSDFDecoder
    decoder = decoder_class(latent_dim=64, hidden_dim=512, num_layers=8,
                            skip_connection=[4], weight_norm=True).to(device)
    decoder.load_state_dict(checkpoint['decoder_state_dict'], strict=True)
    decoder.eval()
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)
    return latent, species, bbox, phylo, matrix, metadata, decoder, manifest_sha


def normalize_real(codes, species, bbox):
    codes_t = torch.as_tensor(codes, dtype=torch.float32)
    species_t = torch.as_tensor(species, dtype=torch.long)
    centroids = torch.stack([codes_t[species_t == i].mean(0) for i in range(24)])
    residuals = codes_t - centroids[species_t]
    bb = torch.as_tensor(bbox, dtype=torch.float32)
    return {'centroids': centroids.numpy(), 'residual_mean': residuals.mean(0).numpy(),
            'residual_global_std': np.asarray(residuals.std().item() + 1e-8, dtype=np.float32),
            'full_code_mean': codes_t.mean(0).numpy(),
            'full_code_global_std': np.asarray(codes_t.std().item() + 1e-8, dtype=np.float32),
            'bbox_mean': bb.mean(0).numpy(), 'bbox_std': (bb.std(0) + 1e-8).numpy()}


def model_kwargs(phylo_dim):
    return {**PROTOCOL['architecture'], 'phylo_embed_dim': int(phylo_dim)}


def prepare(base, work, pair='best', seed=42, device='cuda'):
    need(seed in (42, 43, 44), 'Use predeclared training seeds42/43/44')
    deterministic()
    path = Path(work) / pair / ('seed_' + str(seed)) / 'prepared'
    if (path / 'PREPARED.json').exists():
        load_prepared(path)
        return path
    need(not path.exists() or not any(path.iterdir()), 'Refusing to replace an incomplete/different prepared cache')
    path.mkdir(parents=True, exist_ok=True)
    with isolated_rng():
        seed_all(seed)
        codes, species, bbox, phylo, matrix, metadata, decoder, input_sha = load_legacy(base, pair, device)
        stats = normalize_real(codes, species, bbox)
        # Original source is called with the verified corresponding decoder.
        # An exception aborts preparation; PCA is never substituted.
        augmented, aug_species, aug_bbox, aug_labels = load_vendor('augmentation').augment_latent_codes(
            codes, species, matrix, bbox, PROTOCOL['augmentation'], decoder=decoder, device=device)
        lower, upper = codes.min(0) - 3 * codes.std(0), codes.max(0) + 3 * codes.std(0)
        augmented = np.clip(augmented, lower, upper)
        need(len(augmented) >= 100 and np.array_equal(augmented[:100], codes) and
             np.array_equal(aug_species[:100], species) and np.array_equal(aug_bbox[:100], bbox),
             'Augmentation must preserve original100 records in their original order')
        need(np.isfinite(augmented).all() and np.isfinite(aug_bbox).all(), 'Nonfinite augmented data')
        need(np.array_equal(aug_labels[:100], np.asarray(['original'] * 100)), 'Original item labels differ')
        arrays = {'real_codes': codes, 'real_species': species, 'real_bbox': bbox,
                  'augmented_codes': augmented, 'augmented_species': aug_species,
                  'augmented_bbox': aug_bbox, 'augmented_labels': aug_labels,
                  'phylo_embeddings': phylo, 'historical_taxonomy_proxy': matrix, **stats}
        np.savez_compressed(path / 'DATA.npz', **arrays)
        # Initialization uses an isolated seed and never depends on augmentation draws.
        seed_all(seed)
        net = load_vendor('diffusion').PhyloFlowNet(**model_kwargs(phylo.shape[1]))
        initial = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        initial_sha = state_sha(initial)
        atomic_torch_save({'schema': 'matched_cfm_shared_initialization_v1', 'model': initial,
                           'initial_state_sha256': initial_sha, 'kwargs': model_kwargs(phylo.shape[1])}, path / 'INITIAL.pt')
        # One frozen generation schedule and noise array is reused by every arm.
        eval_species, eval_cohort, eval_slot = [], [], []
        for name in PROTOCOL['generation']['conditional_species']:
            for slot in range(10):
                eval_species.append(metadata['species_to_idx'][name]); eval_cohort.append('conditional_' + name); eval_slot.append(slot)
        for slot in range(100):
            eval_species.append(slot % 24); eval_cohort.append('unconditional_species_mixture'); eval_slot.append(slot)
        eval_species = np.asarray(eval_species, dtype=np.int64)
        real_species_mean_bbox = np.stack([bbox[species == i].mean(0) for i in range(24)])
        eval_bbox = real_species_mean_bbox[eval_species]
        noise_generator = torch.Generator(device='cpu').manual_seed(seed + 100000)
        noise = torch.randn((len(eval_species), 64), generator=noise_generator).numpy()
        np.savez_compressed(path / 'GENERATION_INPUTS.npz', species=eval_species, cohort=np.asarray(eval_cohort),
                            sample_slot=np.asarray(eval_slot), size_physical=eval_bbox,
                            size_normalized=(eval_bbox - stats['bbox_mean']) / stats['bbox_std'],
                            phylo=phylo[eval_species], initial_noise=noise)
    info = {'schema': 'matched_cfm_prepared_v1', 'pair': pair, 'seed': seed,
            'input_manifest_sha256': input_sha, 'source_closure': source_closure(), 'protocol': PROTOCOL,
            'initial_state_sha256': initial_sha, 'augmentation_seed': seed,
            'real_codes': 100, 'augmented_total_codes': len(augmented),
            'original_sampling_probability_when_augmented': 300 / (len(augmented) + 200),
            'original_sampling_probability_no_augmentation': 1.,
            'augmentation_labels': {str(x): int((aug_labels == x).sum()) for x in np.unique(aug_labels)},
            'files': {p.name: file_sha(p) for p in path.iterdir() if p.is_file()}}
    atomic_json(path / 'PREPARED.json', info)
    print(json.dumps({'phase': 'prepared', 'path': str(path), 'pair': pair, 'seed': seed,
                      'augmented_total_codes': len(augmented), 'initial_state_sha256': initial_sha}), flush=True)
    return path


def load_prepared(path):
    path = Path(path)
    info = json.loads((path / 'PREPARED.json').read_text())
    need(info['schema'] == 'matched_cfm_prepared_v1' and info['protocol'] == PROTOCOL and
         info['source_closure'] == source_closure(), 'Prepared protocol/source closure changed')
    for name, digest in info['files'].items():
        need(file_sha(path / name) == digest, 'Prepared cache changed: ' + name)
    with np.load(path / 'DATA.npz', allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    initial = verified_torch_load(path / 'INITIAL.pt')
    need(state_sha(initial['model']) == info['initial_state_sha256'] == initial['initial_state_sha256'],
         'Shared initialization differs')
    return info, arrays, initial


def training_tensors(arrays, variant):
    cfg = VARIANTS[variant]
    prefix = 'augmented' if cfg['augmentation'] else 'real'
    codes = np.asarray(arrays[prefix + '_codes'], dtype=np.float32)
    species = np.asarray(arrays[prefix + '_species'], dtype=np.int64)
    bbox = np.asarray(arrays[prefix + '_bbox'], dtype=np.float32)
    raw = codes - arrays['centroids'][species] if cfg['target'] == 'residual' else codes
    mean, std = arrays[cfg['target'] + '_mean'], arrays[cfg['target'] + '_global_std']
    target = (raw - mean) / std
    recovered = target * std + mean
    if cfg['target'] == 'residual':
        recovered = recovered + arrays['centroids'][species]
    need(np.allclose(recovered, codes, atol=1e-7, rtol=1e-6), 'Target normalization/centroid round-trip failed')
    weights = np.ones(len(codes), dtype=np.float64)
    weights[:100] = 3.
    phylo = np.asarray(arrays['phylo_embeddings'][species], dtype=np.float32)
    if not cfg['phylo_conditioning']:
        phylo = np.zeros_like(phylo)
    values = {'target': target, 'species': species,
              'bbox': (bbox - arrays['bbox_mean']) / arrays['bbox_std'], 'phylo': phylo}
    return {k: torch.as_tensor(v) for k, v in values.items()}, torch.as_tensor(weights)


def train(prepared, out, variant='full', updates=100, device='cuda', checkpoint_every=100,
          stop_after=None):
    need(variant in VARIANTS and updates in (100, 10000), 'Only six predeclared arms and100/10000update budgets allowed')
    need(int(checkpoint_every) > 0, 'Checkpoint interval must be positive')
    deterministic()
    info, arrays, initial = load_prepared(prepared)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    lock = {'schema': 'matched_cfm_training_lock_v1', 'prepared_sha256': file_sha(Path(prepared) / 'PREPARED.json'),
            'variant': variant, 'variant_config': VARIANTS[variant], 'updates': updates,
            'phase': 'pilot_execution_validation_only' if updates == 100 else 'full_fixed_update_study',
            'source_closure': source_closure(), 'protocol': PROTOCOL,
            'pair': info['pair'], 'seed': info['seed'], 'device': device,
            'runtime': runtime_identity(device),
            'initial_state_sha256': info['initial_state_sha256']}
    lock_path = out / 'TRAINING_LOCK.json'
    if lock_path.exists():
        need(json.loads(lock_path.read_text()) == lock, 'Resume protocol, data, source or device differs')
    else:
        need(not any(out.iterdir()), 'Use an empty independent arm output folder')
        atomic_json(lock_path, lock)
    seed_all(info['seed'])
    model = load_vendor('diffusion').PhyloFlowNet(**initial['kwargs']).to(device)
    model.load_state_dict(initial['model'], strict=True)
    need(state_sha(model.state_dict()) == info['initial_state_sha256'], 'Paired initial parameters differ')
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0001, weight_decay=.00001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=updates, eta_min=.000001)
    tensors, weights = training_tensors(arrays, variant)
    sampler = torch.Generator(device='cpu').manual_seed(info['seed'] + 200000)
    flow = torch.Generator(device=device).manual_seed(info['seed'] + 300000)
    completed, trace = 0, []
    total_real = total_augmented = 0
    if (out / 'LATEST.json').exists():
        latest = json.loads((out / 'LATEST.json').read_text())
        path = out / latest['checkpoint']
        need(file_sha(path) == latest['sha256'], 'Latest checkpoint record differs')
        saved = verified_torch_load(path)
        need(saved['lock'] == lock and saved['completed_updates'] == latest['completed_updates'], 'Checkpoint identity differs')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler'])
        completed, trace = saved['completed_updates'], saved['trace']
        total_real, total_augmented = saved['sampled_real'], saved['sampled_augmented']
        restore_rng(saved['rng'], sampler)
        flow.set_state(saved['flow_generator_state'])
    else:
        # Model reconstruction and optimizer setup must not alter the training RNG.
        seed_all(info['seed'] + 400000)
    start = time.perf_counter()
    target_stop = updates if stop_after is None else min(int(stop_after), updates)
    need(target_stop >= completed, 'Requested stop precedes saved update')
    cfg = VARIANTS[variant]
    model.train()
    for step in range(completed + 1, target_stop + 1):
        indices = torch.multinomial(weights, 256, replacement=True, generator=sampler)
        batch = {k: value[indices].to(device) for k, value in tensors.items()}
        x1, species, bbox, phylo = batch['target'], batch['species'], batch['bbox'], batch['phylo']
        dropped = bool(torch.rand((), device=device, generator=flow).item() < .1)
        if dropped:
            species = torch.full_like(species, 24)
        t = torch.rand((256,), device=device, generator=flow)
        x0 = torch.randn(x1.shape, device=device, dtype=x1.dtype, generator=flow)
        xt = (1 - t[:, None]) * x0 + t[:, None] * x1
        predicted = model(xt, t, species, bbox, phylo)
        flow_loss = torch.nn.functional.mse_loss(predicted, x1 - x0)
        mid = .5 * x0 + .5 * x1
        mid_velocity = model(mid, torch.full((256,), .5, device=device), species, bbox, phylo)
        estimate = mid + .5 * mid_velocity
        variance_loss = torch.relu(x1.var() - estimate.var())
        loss = flow_loss + cfg['var_weight'] * variance_loss
        need(bool(torch.isfinite(loss)), 'Nonfinite loss; no checkpoint for this update')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        need(bool(torch.isfinite(norm)), 'Nonfinite gradient; no checkpoint for this update')
        optimizer.step(); scheduler.step()
        n_real = int((indices < 100).sum())
        total_real += n_real; total_augmented += 256 - n_real
        trace.append({'update': step, 'loss': float(loss.detach().cpu()),
                      'flow_loss': float(flow_loss.detach().cpu()), 'variance_loss': float(variance_loss.detach().cpu()),
                      'sampled_real': n_real, 'species_dropped': dropped,
                      'indices_sha256': hashlib.sha256(indices.numpy().tobytes()).hexdigest(),
                      'flow_input_sha256': hashlib.sha256(t.detach().cpu().numpy().tobytes() + x0.detach().cpu().numpy().tobytes()).hexdigest(),
                      'lr': scheduler.get_last_lr()[0]})
        if step % checkpoint_every == 0 or step == target_stop:
            checkpoint = out / f'update_{step:09d}.pt'
            payload = {'schema': 'matched_cfm_exact_resume_v1', 'lock': lock, 'completed_updates': step,
                       'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                       'rng': capture_rng(sampler), 'flow_generator_state': flow.get_state(),
                       'trace': trace, 'sampled_real': total_real, 'sampled_augmented': total_augmented}
            digest = atomic_torch_save(payload, checkpoint)
            atomic_json(out / 'LATEST.json', {'checkpoint': checkpoint.name, 'sha256': digest, 'completed_updates': step})
            atomic_json(out / 'STATUS.json', {'phase': lock['phase'], 'pair': info['pair'], 'seed': info['seed'],
                        'variant': variant, 'completed_updates': step, 'budget_updates': updates,
                        'sampled_real': total_real, 'sampled_augmented': total_augmented,
                        'last_loss': trace[-1]['loss'], 'elapsed_seconds_this_call': time.perf_counter() - start})
            print(json.dumps(json.loads((out / 'STATUS.json').read_text())), flush=True)
    if target_stop == updates:
        atomic_json(out / 'TRAINING_COMPLETE.json', {'schema': 'matched_cfm_fixed_budget_complete_v1',
                    'lock_sha256': file_sha(lock_path), 'completed_updates': updates,
                    'model_sha256': state_sha(model.state_dict()), 'latest': json.loads((out / 'LATEST.json').read_text()),
                    'phase': lock['phase'], 'performance_claim': False})
    return {'out': str(out), 'completed_updates': target_stop, 'model_sha256': state_sha(model.state_dict())}


@torch.no_grad()
def sample(prepared, trained, out, device='cuda'):
    info, arrays, initial = load_prepared(prepared)
    trained, out = Path(trained), Path(out)
    need((trained / 'TRAINING_COMPLETE.json').exists(), 'Sampling requires completed predeclared update budget')
    latest = json.loads((trained / 'LATEST.json').read_text())
    saved = verified_torch_load(trained / latest['checkpoint'])
    need(file_sha(trained / latest['checkpoint']) == latest['sha256'], 'Sampling checkpoint record differs')
    lock = json.loads((trained / 'TRAINING_LOCK.json').read_text())
    need(lock == saved['lock'] and lock['prepared_sha256'] == file_sha(Path(prepared) / 'PREPARED.json') and
         lock['source_closure'] == source_closure(), 'Sampling model/source/data differs')
    complete = json.loads((trained / 'TRAINING_COMPLETE.json').read_text())
    need(complete.get('schema') == 'matched_cfm_fixed_budget_complete_v1' and
         complete.get('lock_sha256') == file_sha(trained / 'TRAINING_LOCK.json') and
         complete.get('completed_updates') == saved['completed_updates'] == lock['updates'] and
         complete.get('latest') == latest and complete.get('model_sha256') == state_sha(saved['model']),
         'Sampling completion seal does not match this final fixed-budget checkpoint')
    need(not out.exists() or not any(out.iterdir()), 'Use a new independent generation output folder')
    out.mkdir(parents=True, exist_ok=True)
    with isolated_rng():
        model = load_vendor('diffusion').PhyloFlowNet(**initial['kwargs']).to(device)
        model.load_state_dict(saved['model']); model.eval()
        with np.load(Path(prepared) / 'GENERATION_INPUTS.npz', allow_pickle=False) as data:
            generation = {k: data[k] for k in data.files}
        cfg = VARIANTS[lock['variant']]
        species = torch.as_tensor(generation['species'], device=device)
        bbox = torch.as_tensor(generation['size_normalized'], device=device, dtype=torch.float32)
        phylo = torch.as_tensor(generation['phylo'], device=device, dtype=torch.float32)
        if not cfg['phylo_conditioning']:
            phylo = torch.zeros_like(phylo)
        x = torch.as_tensor(generation['initial_noise'], device=device, dtype=torch.float32).clone()
        x *= PROTOCOL['generation']['temperature']
        for step in range(50):
            t = torch.full((len(x),), step / 50., device=device)
            conditional = model(x, t, species, bbox, phylo)
            unconditional_species = model(x, t, None, bbox, phylo)
            x += (unconditional_species + 1.5 * (conditional - unconditional_species)) / 50.
        target = x.cpu().numpy()
        codes = target * arrays[cfg['target'] + '_global_std'] + arrays[cfg['target'] + '_mean']
        if cfg['target'] == 'residual':
            codes = codes + arrays['centroids'][generation['species']]
        need(codes.shape == (140, 64) and np.isfinite(codes).all(), 'Nonfinite or incomplete generated codes')
        np.savez_compressed(out / 'GENERATED_CODES.npz', codes=codes, normalized_targets=target,
                            species=generation['species'], cohort=generation['cohort'], sample_slot=generation['sample_slot'])
    atomic_json(out / 'GENERATION_MANIFEST.json', {'schema': 'matched_cfm_generated_codes_v1',
                'pair': info['pair'], 'seed': info['seed'], 'variant': lock['variant'], 'phase': lock['phase'],
                'completed_updates': saved['completed_updates'], 'source_closure': source_closure(),
                'prepared_sha256': file_sha(Path(prepared) / 'PREPARED.json'),
                'generation_inputs_sha256': file_sha(Path(prepared) / 'GENERATION_INPUTS.npz'),
                'model_checkpoint_sha256': latest['sha256'], 'codes_sha256': file_sha(out / 'GENERATED_CODES.npz'),
                'generation_protocol': PROTOCOL['generation'], 'decoder_meshes_generated': False,
                'performance_claim': False})
    return str(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'train', 'sample'])
    parser.add_argument('--base', default=str(HERE))
    parser.add_argument('--work')
    parser.add_argument('--pair', default='best', choices=['best', 'final'])
    parser.add_argument('--seed', type=int, default=42, choices=[42, 43, 44])
    parser.add_argument('--prepared')
    parser.add_argument('--trained')
    parser.add_argument('--out')
    parser.add_argument('--variant', choices=list(VARIANTS), default='full')
    parser.add_argument('--updates', type=int, choices=[100, 10000], default=100)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--checkpoint-every', type=int, default=100)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    need(args.threads > 0, 'Thread count must be positive')
    torch.set_num_threads(args.threads)
    if args.action == 'prepare':
        need(args.work, '--work required')
        result = prepare(args.base, args.work, args.pair, args.seed, args.device)
    elif args.action == 'train':
        need(args.prepared and args.out, '--prepared and --out required')
        result = train(args.prepared, args.out, args.variant, args.updates, args.device, args.checkpoint_every)
    else:
        need(args.prepared and args.trained and args.out, '--prepared --trained --out required')
        result = sample(args.prepared, args.trained, args.out, args.device)
    print(json.dumps({'result': str(result)}), flush=True)


if __name__ == '__main__':
    main()
