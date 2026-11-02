# READABLE REFERENCE COPY — PhyloSDF_shuffled_matrix_control_v2.ipynb, zero-based cell 3.
# Scientific function bodies are preserved exactly; only notebook magic and listed binary payload assignments were removed.
# Full reproduction requires original notebook, external source archives, input data and checkpoints.
# shuf_engine.py -- PhyloSDF shuffled-matrix (permutation) control for the revision.
#
# Built from rep_engine.py (the loss-ablation / held-out engine whose runs produced Table 5) with three changes:
#   1. a new study 'shuffle_control': the DeepSDF auto-decoder is retrained on all 100 skulls with the paper's loss
#      (beta = 5e-3) but with the proxy distance matrix's species labels permuted (fixed permutations, embedded and
#      checksummed below). Everything else is identical to the 'beta_proxy' arm: recipe, initial weights, data order,
#      reconstruction and generation with the final model's pooled residual flow, frozen metrics.
#   2. the latent summary also reports agreement with the matrix actually used in training ('used').
#   3. the Chamfer nearest-neighbour search uses the frozen evaluator's exact KD-tree on all CPU cores (as in the
#      final-model v2 engine) instead of the slow GPU kernel; values are unchanged (validation check 1).
#
# Original description of rep_engine.py follows.
#
# Google Drive is used READ-ONLY: this code only downloads files by ID. It never uploads, edits, moves,
# deletes or shares anything. All outputs stay in /content and are downloaded to your computer as zips.
#
# Study 'loss_ablation' (Table 5): the DeepSDF auto-decoder is retrained on all 100 skulls three times from
#   identical initial weights and identical data order, changing only the phylogenetic consistency loss:
#     beta0        beta = 0 (no phylogenetic loss)
#     beta_proxy   beta = 5e-3 with the genus-group proxy matrix used in the paper
#     beta_tree    beta = 5e-3 with patristic distances from the dated tree (McTavish et al. 2025)
#   Each representation: latent geometry, reconstruction of the 100 skulls at 256^3, and generation with the
#   pooled residual flow (the final model; 140 skulls), all scored with the frozen metric code.
#
# Study 'heldout' (replaces leave-one-species-out): the auto-decoder is retrained on the 79 skulls of 20 species;
#   the 21 skulls of the 4 focal species are never seen in representation learning. Afterwards the decoder is
#   frozen and the withheld skulls are fitted; each withheld species' centroid is predicted from its relatives
#   (proxy, dated tree, or no phylogeny), skulls are generated for the unseen species and scored.
#     ho_beta_proxy  beta = 5e-3, proxy matrix (the paper's setting)
#     ho_beta0       beta = 0
#     ho_beta_tree   beta = 5e-3, dated tree (optional, runs last)
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys, json, time, hashlib, zipfile, shutil, importlib.util, importlib.metadata, datetime
from pathlib import Path
import numpy as np

ROOT = Path('/content/REP')
DL = ROOT / 'downloads'
SRC = ROOT / 'src' / 'matched_cfm'
MESHSRC = ROOT / 'meshsrc' / 'new_mesh_metrics'
REFS = ROOT / 'refs'
WORK = ROOT / 'work'
OUT = ROOT / 'results'
SURFACE_SEEDS = (20261014, 20261015)
FOCAL = ('G.Fortis', 'G.Magnirostris', 'G.Fuliginosa', 'C.Parvulus')
FINAL_DECODER_SHA = 'd73acbdd6bd2836a352ad3e7434699292577492f813c9f305769115362cc69c7'
SEED = 42                      # DeepSDF initialisation / subsampling seed (as in the original config)
RES = 256                      # decoding grid (endpoint-inclusive, [-1, 1]^3)
TRAIN_TF32 = True              # TF32 matrix multiplies for DeepSDF training only (decoding/metrics stay FP32)
GEN_SEEDS = (42,)              # residual-flow training seeds for the generation part
STUDY_GROUP = 'representation_studies_v1'
HKT = datetime.timezone(datetime.timedelta(hours=8))

INPUTS = {
    'matched_cfm_source_and_inputs_20261018_v1.zip': (
        ['local-input-4547bb9fdfe0aaff', 'local-input-ec3acf21ef60d480'],
        '93f2dc31651f900981076010a6a039bd84314434c41270fbdf64e97551cef169'),
    'new_matched_mesh800_source_20261018_v1.zip': (
        ['local-input-84ff5aba929ef1c4', 'local-input-b634c9ff5803ca39'],
        '30e17ce1efa4a478e5253b29f9885f4b2f3999a1b1ea27599550dc2220577eb7'),
}

# Original DeepSDF recipe (config.yaml of the paper; 02_train_deepsdf.py).
DEEPSDF = {'latent_dim': 64, 'hidden_dim': 512, 'num_layers': 8, 'skip_connection': [4], 'weight_norm': True,
           'latent_reg_weight': 1e-3, 'phylo_reg_weight': 5e-3, 'clamping_distance': 0.1, 'lr_decoder': 5e-4,
           'lr_latent': 1e-3, 'epochs': 2000, 'batch_size': 16384, 'checkpoint_every': 200,
           'lr_schedule_step': 500, 'lr_schedule_gamma': 0.5}
SAMPLES_PER_SPECIMEN = 50000   # SDFDataset(subsample=50000), drawn once with np.random after seeding
PHYLO_EVERY = 10               # phylogenetic loss every 10th batch of an epoch (batch 0, 10, 20, ...)

STUDIES = {
    'loss_ablation': {
        'beta0': {'beta': 0.0, 'matrix': 'proxy', 'specimens': 'all'},
        'beta_proxy': {'beta': 5e-3, 'matrix': 'proxy', 'specimens': 'all'},
        'beta_tree': {'beta': 5e-3, 'matrix': 'tree', 'specimens': 'all'},
    },
    'shuffle_control': {f'beta_shuf{k}': {'beta': 5e-3, 'matrix': f'proxy_shuf{k}', 'specimens': 'all'}
                        for k in range(1, 6)},
    'heldout': {
        'ho_beta_proxy': {'beta': 5e-3, 'matrix': 'proxy', 'specimens': 'train79'},
        'ho_beta0': {'beta': 0.0, 'matrix': 'proxy', 'specimens': 'train79'},
        'ho_beta_tree': {'beta': 5e-3, 'matrix': 'tree', 'specimens': 'train79'},
    },
}
METHOD_NAMES = ('rep_pooled_residual_cfm',
                'ho_gen_proxy_knn', 'ho_gen_tree_knn', 'ho_gen_global_mean', 'ho_gen_oracle',
                'ho_centroid_proxy_knn', 'ho_centroid_tree_knn', 'ho_centroid_global_mean', 'ho_centroid_oracle')
PREDICTORS = ('proxy_knn', 'tree_knn', 'global_mean', 'oracle')
FIT = {'steps': 2000, 'lr': 1e-3, 'points_per_specimen': 4096, 'init_jitter': 0.01}
# Budgets; dry_run() shrinks them to exercise every step of an arm in a few minutes before the real run.
BUDGET = {'epochs': None, 'flow_updates': 10000, 'fit_steps': None, 'rows_per_method': None}

# Patristic distances (Myr) between the 24 species in the dataset's species order, from the dated
# maximum-clade-credibility tree of all birds (McTavish et al. 2025, Clements 2024 taxonomy).
TREE_SPECIES = ['C.Pallidus', 'C.Parvulus', 'C.Psittacula', 'C.flaveola', 'C2.Fusca', 'C2.Olivacea', 'E.campestris',
                'G.Conirostris', 'G.Difficilis', 'G.Fortis', 'G.Fuliginosa', 'G.Magnirostris', 'G.Scandens',
                'G.Septentrionalist', 'L.anoxanthus', 'L.noctis', 'L.portoricensis', 'L.violacea', 'M.nigra',
                'P.Crassirostris', 'P2.Inornata', 'T.Bicolor', 'T.canora', 'T.olivacea']
TREE_MYR = [[0.0,0.482734,0.320264,16.586414,3.163785,2.609423,12.380132,1.457115,1.457115,1.457115,1.457115,1.457115,1.457115,1.457115,9.673585,8.956846,12.380132,12.380131,12.380132,2.055061,1.164521,7.574101,8.488243,14.26648],[0.482734,0.0,0.482734,16.586414,3.163785,2.609423,12.380132,1.457115,1.457115,1.457115,1.457115,1.457115,1.457115,1.457115,9.673585,8.956846,12.380132,12.380131,12.380132,2.055061,1.164521,7.574101,8.488243,14.26648],[0.320264,0.482734,0.0,16.586414,3.163785,2.609423,12.380132,1.457115,1.457115,1.457115,1.457115,1.457115,1.457115,1.457115,9.673585,8.956846,12.380132,12.380131,12.380132,2.055061,1.164521,7.574101,8.488243,14.26648],[16.586414,16.586414,16.586414,0.0,16.586415,16.586415,16.586416,16.586415,16.586415,16.586415,16.586415,16.586415,16.586415,16.586415,16.586415,16.586414,16.586416,16.586415,16.586416,16.586415,16.586415,16.586415,16.586415,16.586416],[3.163785,3.163785,3.163785,16.586415,0.0,3.163786,12.380133,3.163786,3.163786,3.163786,3.163786,3.163786,3.163786,3.163786,9.673586,8.956847,12.380133,12.380132,12.380133,3.163786,3.163786,7.574102,8.488244,14.266481],[2.609423,2.609423,2.609423,16.586415,3.163786,0.0,12.380133,2.609424,2.609424,2.609424,2.609424,2.609424,2.609424,2.609424,9.673586,8.956847,12.380133,12.380132,12.380133,2.609424,2.609424,7.574102,8.488244,14.266481],[12.380132,12.380132,12.380132,16.586416,12.380133,12.380133,0.0,12.380133,12.380133,12.380133,12.380133,12.380133,12.380133,12.380133,12.380133,12.380132,11.206292,11.206291,11.206292,12.380133,12.380133,12.380133,12.380133,14.266482],[1.457115,1.457115,1.457115,16.586415,3.163786,2.609424,12.380133,0.0,0.81275,0.652024,0.81275,0.192978,0.652024,0.81275,9.673586,8.956847,12.380133,12.380132,12.380133,2.055062,1.457116,7.574102,8.488244,14.266481],[1.457115,1.457115,1.457115,16.586415,3.163786,2.609424,12.380133,0.81275,0.0,0.81275,0.541836,0.81275,0.81275,0.270922,9.673586,8.956847,12.380133,12.380132,12.380133,2.055062,1.457116,7.574102,8.488244,14.266481],[1.457115,1.457115,1.457115,16.586415,3.163786,2.609424,12.380133,0.652024,0.81275,0.0,0.81275,0.652024,0.525484,0.81275,9.673586,8.956847,12.380133,12.380132,12.380133,2.055062,1.457116,7.574102,8.488244,14.266481],[1.457115,1.457115,1.457115,16.586415,3.163786,2.609424,12.380133,0.81275,0.541836,0.81275,0.0,0.81275,0.81275,0.541836,9.673586,8.956847,12.380133,12.380132,12.380133,2.055062,1.457116,7.574102,8.488244,14.266481],[1.457115,1.457115,1.457115,16.586415,3.163786,2.609424,12.380133,0.192978,0.81275,0.652024,0.81275,0.0,0.652024,0.81275,9.673586,8.956847,12.380133,12.380132,12.380133,2.055062,1.457116,7.574102,8.488244,14.266481],[1.457115,1.457115,1.457115,16.586415,3.163786,2.609424,12.380133,0.652024,0.81275,0.525484,0.81275,0.652024,0.0,0.81275,9.673586,8.956847,12.380133,12.380132,12.380133,2.055062,1.457116,7.574102,8.488244,14.266481],[1.457115,1.457115,1.457115,16.586415,3.163786,2.609424,12.380133,0.81275,0.270922,0.81275,0.541836,0.81275,0.81275,0.0,9.673586,8.956847,12.380133,12.380132,12.380133,2.055062,1.457116,7.574102,8.488244,14.266481],[9.673585,9.673585,9.673585,16.586415,9.673586,9.673586,12.380133,9.673586,9.673586,9.673586,9.673586,9.673586,9.673586,9.673586,0.0,9.673585,12.380133,12.380132,12.380133,9.673586,9.673586,9.673586,9.673586,14.266481],[8.956846,8.956846,8.956846,16.586414,8.956847,8.956847,12.380132,8.956847,8.956847,8.956847,8.956847,8.956847,8.956847,8.956847,9.673585,0.0,12.380132,12.380131,12.380132,8.956847,8.956847,8.956847,8.956847,14.26648],[12.380132,12.380132,12.380132,16.586416,12.380133,12.380133,11.206292,12.380133,12.380133,12.380133,12.380133,12.380133,12.380133,12.380133,12.380133,12.380132,0.0,9.956233,9.956234,12.380133,12.380133,12.380133,12.380133,14.266482],[12.380131,12.380131,12.380131,16.586415,12.380132,12.380132,11.206291,12.380132,12.380132,12.380132,12.380132,12.380132,12.380132,12.380132,12.380132,12.380131,9.956233,0.0,8.622545,12.380132,12.380132,12.380132,12.380132,14.266481],[12.380132,12.380132,12.380132,16.586416,12.380133,12.380133,11.206292,12.380133,12.380133,12.380133,12.380133,12.380133,12.380133,12.380133,12.380133,12.380132,9.956234,8.622545,0.0,12.380133,12.380133,12.380133,12.380133,14.266482],[2.055061,2.055061,2.055061,16.586415,3.163786,2.609424,12.380133,2.055062,2.055062,2.055062,2.055062,2.055062,2.055062,2.055062,9.673586,8.956847,12.380133,12.380132,12.380133,0.0,2.055062,7.574102,8.488244,14.266481],[1.164521,1.164521,1.164521,16.586415,3.163786,2.609424,12.380133,1.457116,1.457116,1.457116,1.457116,1.457116,1.457116,1.457116,9.673586,8.956847,12.380133,12.380132,12.380133,2.055062,0.0,7.574102,8.488244,14.266481],[7.574101,7.574101,7.574101,16.586415,7.574102,7.574102,12.380133,7.574102,7.574102,7.574102,7.574102,7.574102,7.574102,7.574102,9.673586,8.956847,12.380133,12.380132,12.380133,7.574102,7.574102,0.0,8.488244,14.266481],[8.488243,8.488243,8.488243,16.586415,8.488244,8.488244,12.380133,8.488244,8.488244,8.488244,8.488244,8.488244,8.488244,8.488244,9.673586,8.956847,12.380133,12.380132,12.380133,8.488244,8.488244,8.488244,0.0,14.266481],[14.26648,14.26648,14.26648,16.586416,14.266481,14.266481,14.266482,14.266481,14.266481,14.266481,14.266481,14.266481,14.266481,14.266481,14.266481,14.26648,14.266482,14.266481,14.266482,14.266481,14.266481,14.266481,14.266481,0.0]]
TREE_SHA = 'ab1cb983b3d0a3d527c0f3156961c4d1f3985db59b7117b2b40e07cfb04b4141'

# Species-label permutations of the proxy matrix for the shuffled-matrix control. Drawn once with
# numpy.random.default_rng(20261020).permutation(24), keeping the first five that move every species (no fixed
# point) and whose permuted matrix is essentially uncorrelated with both the proxy and the dated tree
# (|Pearson r| < 0.1 over the 276 species pairs; 92% of random permutations meet this). Shuffled matrix k:
# D_k[i, j] = proxy[p_k[i], p_k[j]] -- the same distance values and block structure, assigned to the wrong species.
SHUFFLES = {"1": [8, 15, 1, 16, 19, 3, 14, 0, 5, 20, 11, 12, 9, 6, 2, 18, 21, 13, 4, 22, 17, 10, 23, 7], "2": [18, 6, 16, 19, 8, 7, 0, 12, 1, 13, 3, 15, 21, 22, 11, 2, 4, 20, 17, 5, 9, 10, 23, 14], "3": [23, 22, 0, 7, 6, 14, 18, 8, 17, 21, 13, 1, 9, 20, 15, 2, 3, 16, 4, 10, 5, 11, 12, 19], "4": [1, 11, 17, 4, 13, 19, 20, 12, 6, 8, 2, 9, 21, 18, 7, 10, 14, 16, 15, 3, 0, 22, 23, 5], "5": [1, 0, 10, 16, 14, 13, 5, 9, 21, 15, 7, 3, 8, 19, 11, 20, 6, 18, 2, 22, 12, 23, 4, 17]}
SHUFFLE_R = {"1": {"r_with_proxy": -0.041546, "r_with_tree": -0.027714}, "2": {"r_with_proxy": -0.064781, "r_with_tree": 0.018872}, "3": {"r_with_proxy": -0.030433, "r_with_tree": -0.040641}, "4": {"r_with_proxy": 0.014522, "r_with_tree": 0.095158}, "5": {"r_with_proxy": -0.030938, "r_with_tree": 0.087876}}
SHUFFLES_SHA = '40d2912bfcebcc4f629d95acdcf3a967f23bbe7544e064316e932b94aeed28a0'

# Frozen per-specimen reconstruction CDs of the paper's final decoder (primary 256^3 study).
FROZEN_RECON = {'C.PallidusA': 0.0002330920312383047, 'C.PallidusB': 0.00020308618230866851}


def log(msg):
    print(f"[{datetime.datetime.now(HKT):%m-%d %H:%M:%S} HKT] {msg}", flush=True)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(_plain(value), indent=1))
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
        raise RuntimeError('No GPU. Runtime > Change runtime type > GPU (A100 or L4), then run again.')
    for d in (DL, WORK, OUT):
        d.mkdir(parents=True, exist_ok=True)
    log('downloading inputs (read-only)...')
    with zipfile.ZipFile(get_input('matched_cfm_source_and_inputs_20261018_v1.zip')) as z:
        if not (SRC / 'run_matched_cfm.py').exists():
            z.extractall(ROOT / 'src')
    with zipfile.ZipFile(get_input('new_matched_mesh800_source_20261018_v1.zip')) as z:
        if not (MESHSRC / 'vendor' / 'vendor_metrics.py').exists():
            z.extractall(ROOT / 'meshsrc')
    if sha(SRC / 'inputs' / 'final.pt') != FINAL_DECODER_SHA:
        raise RuntimeError('final.pt checksum differs from the paper decoder')
    manifest = MESHSRC / 'vendor' / 'reference_manifest.json'
    rows = json.loads(manifest.read_text())['references']
    log(f'downloading the {len(rows)} real skulls (watertight STL + SDF samples, read-only)...')
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
            log(f'  skulls {i + 1}/{len(rows)}')
    shutil.copy(manifest, REFS / 'reference_manifest.json')
    if hashlib.sha256(json.dumps(TREE_MYR, separators=(',', ':')).encode()).hexdigest() != TREE_SHA:
        raise RuntimeError('embedded dated-tree matrix changed')
    env = {'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
           'numpy': np.__version__, 'python': sys.version.split()[0], 'train_tf32': TRAIN_TF32}
    for name in ('scipy', 'scikit-image', 'trimesh'):
        env[name] = importlib.metadata.version(name)
    write_json(OUT / 'ENVIRONMENT.json', env)
    write_json(OUT / 'SHUFFLES.json', check_shuffles())
    log('environment: ' + json.dumps(env))


# ---------------------------------------------------------------- data and matrices ----------
def metadata():
    return json.loads((SRC / 'inputs' / 'metadata.json').read_text())


def species_names():
    meta = metadata()
    return [meta['idx_to_species'][str(i)] for i in range(meta['num_species'])]


def specimens():
    """Specimen names and species indices in the original training order (metadata order)."""
    meta = metadata()
    names = [s['filename'] for s in meta['specimens']]
    species = np.array([s['species_idx'] for s in meta['specimens']], dtype=np.int64)
    return names, species


def processed_path(name):
    return REFS / 'processed' / f'{name}.npz'


def selection(kind):
    names, species = specimens()
    sp_names = species_names()
    if kind == 'all':
        return list(range(len(names)))
    if kind == 'train79':
        return [i for i in range(len(names)) if sp_names[species[i]] not in FOCAL]
    if kind == 'heldout21':
        return [i for i in range(len(names)) if sp_names[species[i]] in FOCAL]
    raise ValueError(kind)


def matrix(kind):
    if kind == 'proxy':
        return np.load(SRC / 'inputs' / 'phylo_dist.npy').astype(np.float64)
    if kind == 'tree':
        if TREE_SPECIES != species_names():
            raise RuntimeError('species order differs from the embedded tree matrix')
        tree = np.asarray(TREE_MYR, dtype=np.float64)
        return tree / tree.max()          # Pearson correlation is scale-free; normalised for readability
    if kind.startswith('proxy_shuf'):
        p = np.asarray(SHUFFLES[kind[len('proxy_shuf'):]], dtype=np.int64)
        P = matrix('proxy')
        return P[np.ix_(p, p)]
    raise ValueError(kind)


def check_shuffles():
    """The embedded permutations are unchanged, move every species, and give matrices uncorrelated with the proxy and
    the dated tree; returns the record saved with the results."""
    perms = [SHUFFLES[str(k)] for k in range(1, len(SHUFFLES) + 1)]
    if hashlib.sha256(json.dumps(perms, separators=(',', ':')).encode()).hexdigest() != SHUFFLES_SHA:
        raise RuntimeError('embedded shuffles changed')
    proxy, tree = matrix('proxy'), matrix('tree')
    iu = np.triu_indices(len(proxy), 1)
    out = {}
    for k, p in SHUFFLES.items():
        if sorted(p) != list(range(len(proxy))) or any(int(a) == i for i, a in enumerate(p)):
            raise RuntimeError(f'shuffle {k} is not a derangement of the {len(proxy)} species')
        M = matrix('proxy_shuf' + k)
        rp, rt = float(np.corrcoef(M[iu], proxy[iu])[0, 1]), float(np.corrcoef(M[iu], tree[iu])[0, 1])
        if abs(rp - SHUFFLE_R[k]['r_with_proxy']) > 1e-5 or abs(rt - SHUFFLE_R[k]['r_with_tree']) > 1e-5:
            raise RuntimeError(f'shuffle {k}: correlations differ from the embedded record')
        names = species_names()
        out['proxy_shuf' + k] = {'permutation': p, 'r_with_proxy': rp, 'r_with_tree': rt,
                                 'species_takes_proxy_row_of': {names[i]: names[a] for i, a in enumerate(p)}}
    return out


def pair_correlation(codes, species, M):
    from scipy.spatial.distance import pdist
    from scipy.stats import pearsonr
    i, j = np.triu_indices(len(codes), 1)
    return float(pearsonr(pdist(np.asarray(codes, np.float64)), M[species[i], species[j]])[0])


def latent_summary(codes, species, permutations=9999, seed=0, used=None):
    """Specimen-level LPC and centroid-level correlations with the proxy and the dated tree (Mantel p); with
    used='proxy_shufK' also with the matrix used in training."""
    from scipy.spatial.distance import pdist, squareform
    proxy, tree = matrix('proxy'), matrix('tree')
    extra = (('used', matrix(used)),) if used not in (None, 'proxy', 'tree') else ()
    present = np.array(sorted(set(species.tolist())))
    C = np.stack([codes[species == s].mean(0) for s in present])
    L = squareform(pdist(np.asarray(C, np.float64)))
    iu = np.triu_indices(len(present), 1)
    out = {'species': len(present), 'specimens': len(codes),
           'specimen_LPC_proxy': pair_correlation(codes, species, proxy),
           'specimen_corr_tree': pair_correlation(codes, species, tree)}
    if extra:
        out['matrix_used'] = used
        out['specimen_LPC_used'] = pair_correlation(codes, species, extra[0][1])
    rng = np.random.default_rng(seed)
    for label, M in (('proxy', proxy), ('tree', tree)) + extra:
        sub = M[np.ix_(present, present)]
        r = float(np.corrcoef(L[iu], sub[iu])[0, 1])
        count = 0
        for _ in range(permutations):
            p = rng.permutation(len(present))
            count += float(np.corrcoef(L[iu], sub[np.ix_(p, p)][iu])[0, 1]) >= r
        out[f'centroid_corr_{label}'] = r
        out[f'centroid_mantel_p_{label}'] = (count + 1) / (permutations + 1)
    out['permutations'] = permutations
    return out


# ---------------------------------------------------------------- precision modes ------------
def training_mode():
    import torch
    torch.use_deterministic_algorithms(False)
    torch.backends.cuda.matmul.allow_tf32 = bool(TRAIN_TF32)
    torch.backends.cudnn.allow_tf32 = bool(TRAIN_TF32)
    torch.backends.cudnn.benchmark = False


def fp32_mode():
    """FP32 matrix multiplies without the deterministic-algorithms switch (latent fitting)."""
    import torch
    torch.use_deterministic_algorithms(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def exact_mode():
    """FP32 and deterministic, as the frozen decoding/benchmark and the matched CFM protocol."""
    import torch
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision('highest')


_vd = None


def vendor_deepsdf():
    """The original DeepSDF module of the paper (decoder, latent library, losses), unchanged."""
    global _vd
    if _vd is None:
        _vd = load_module(SRC / 'vendor' / 'deepsdf.py', 'rep_vendor_deepsdf')
    return _vd


def state_digest(*states):
    h = hashlib.sha256()
    for state in states:
        for name, tensor in sorted(state.items()):
            value = tensor.detach().cpu().contiguous()
            h.update(name.encode()); h.update(str(tuple(value.shape)).encode()); h.update(value.numpy().tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------- DeepSDF training -----------
def _phylo_loss_fn(n, species, M):
    """Vectorised form of the original phylogenetic_consistency_loss: negative Pearson correlation over all
    N(N-1)/2 specimen pairs. torch.pdist returns pairs in the same row-major order as torch.triu_indices."""
    import torch
    i, j = np.triu_indices(n, 1)
    pd = torch.tensor(M[species[i], species[j]], dtype=torch.float32, device='cuda')
    pd_centered = pd - pd.mean()
    pd_ss = (pd_centered ** 2).sum()

    def loss(W):
        ld = torch.pdist(W)
        ld_centered = ld - ld.mean()
        corr = (ld_centered * pd_centered).sum() / (torch.sqrt((ld_centered ** 2).sum() * pd_ss) + 1e-8)
        return -corr
    return loss


def _check_phylo_equivalence(W0, species, M, fast):
    """Value and gradient of the vectorised loss equal the original loop implementation."""
    import torch
    vd = vendor_deepsdf()

    class Lib:
        pass
    a = W0.detach().clone().requires_grad_(True)
    lib = Lib(); lib.latent_codes = Lib(); lib.latent_codes.weight = a
    original = vd.phylogenetic_consistency_loss(lib, species, M.astype(np.float32))
    original.backward()
    b = W0.detach().clone().requires_grad_(True)
    mine = fast(b)
    mine.backward()
    dv = abs(float(original) - float(mine))
    dg = float((a.grad - b.grad).abs().max())
    scale = float(a.grad.abs().max())
    if dv > 1e-5 or dg > 1e-4 * scale + 1e-9:
        raise RuntimeError(f'phylogenetic loss vectorisation differs: value {dv:.2e}, grad {dg:.2e}')
    return {'value_original': float(original), 'value_vectorised': float(mine), 'abs_value_difference': dv,
            'max_abs_gradient_difference': dg, 'max_abs_gradient': scale}


def save_checkpoint(path, decoder_state, latent_state, epoch, loss, cfg):
    """Same structure as the paper's final.pt, so the frozen loader and decoder accept it."""
    import torch
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    torch.save({'epoch': int(epoch), 'decoder_state_dict': {k: v.detach().cpu() for k, v in decoder_state.items()},
                'latent_lib_state_dict': {k: v.detach().cpu() for k, v in latent_state.items()},
                'loss': float(loss), 'config': dict(cfg)}, tmp)
    tmp.replace(path)


def deepsdf_dir(arm):
    return WORK / arm / 'deepsdf'


def train_deepsdf(arm, spec, epochs=None, keep=None, out=None, log_every=50):
    """Original recipe: 50,000 SDF samples per skull drawn once with np.random (seed 42), batch 16,384 with
    drop_last, Adam (decoder 5e-4, codes 1e-3), StepLR 500 epochs x 0.5, clamped L1 + 1e-3 code penalty, and
    beta * phylogenetic loss every 10th batch. Initial weights and data order are identical for every arm with the
    same specimens (seed 42; the batch order uses its own generator)."""
    import torch
    out = Path(out) if out is not None else deepsdf_dir(arm)
    if (out / 'TRAINING_COMPLETE.json').exists():
        return out
    out.mkdir(parents=True, exist_ok=True)
    E = int(epochs or DEEPSDF['epochs'])
    keep = list(keep) if keep is not None else selection(spec['specimens'])
    names, species_all = specimens()
    species = species_all[keep]
    M = matrix(spec['matrix'])
    beta = float(spec['beta'])
    cfg = dict(DEEPSDF, phylo_reg_weight=beta, epochs=E)
    vd = vendor_deepsdf()
    training_mode()
    torch.manual_seed(SEED); np.random.seed(SEED); torch.cuda.manual_seed_all(SEED)
    P, D, I, picks = [], [], [], []
    for k, i in enumerate(keep):
        with np.load(processed_path(names[i]), allow_pickle=False) as data:
            pts = np.asarray(data['points'], dtype=np.float32)
            sdf = np.asarray(data['sdf'], dtype=np.float32).reshape(-1)
        if len(pts) > SAMPLES_PER_SPECIMEN:
            sel = np.random.choice(len(pts), SAMPLES_PER_SPECIMEN, replace=False)
        else:
            sel = np.arange(len(pts))
        picks.append(sel)
        P.append(pts[sel]); D.append(sdf[sel]); I.append(np.full(len(sel), k, dtype=np.int64))
    data_digest = hashlib.sha256(b''.join(np.asarray(s, np.int64).tobytes() for s in picks)).hexdigest()
    P = torch.as_tensor(np.concatenate(P), device='cuda')
    D = torch.as_tensor(np.concatenate(D), device='cuda')
    I = torch.as_tensor(np.concatenate(I), device='cuda')
    decoder = vd.DeepSDFDecoder(latent_dim=64, hidden_dim=512, num_layers=8, skip_connection=[4],
                                weight_norm=True).to('cuda')
    latent = vd.LatentCodeLibrary(len(keep), 64).to('cuda')
    init_digest = state_digest(decoder.state_dict(), latent.state_dict())
    opt_dec = torch.optim.Adam(decoder.parameters(), lr=cfg['lr_decoder'])
    opt_lat = torch.optim.Adam(latent.parameters(), lr=cfg['lr_latent'])
    sch_dec = torch.optim.lr_scheduler.StepLR(opt_dec, cfg['lr_schedule_step'], cfg['lr_schedule_gamma'])
    sch_lat = torch.optim.lr_scheduler.StepLR(opt_lat, cfg['lr_schedule_step'], cfg['lr_schedule_gamma'])
    phylo = _phylo_loss_fn(len(keep), species, M)
    equivalence = _check_phylo_equivalence(latent.latent_codes.weight, species, M, phylo) if len(keep) >= 3 else None
    order = torch.Generator(device='cuda')
    order.manual_seed(SEED + 1)
    R, B = len(I), int(cfg['batch_size'])
    nbatches = R // B
    if nbatches < 1:
        raise RuntimeError('fewer SDF samples than one batch')
    delta, reg_w = cfg['clamping_distance'], cfg['latent_reg_weight']
    rows, start_epoch, best = [], 1, {'loss': float('inf')}
    resume = out / 'resume.pt'
    if resume.exists():
        saved = torch.load(resume, map_location='cuda', weights_only=False)
        if saved['init_digest'] != init_digest or saved['data_digest'] != data_digest or saved['cfg'] != cfg:
            raise RuntimeError('resume state belongs to a different run; delete ' + str(resume))
        decoder.load_state_dict(saved['decoder']); latent.load_state_dict(saved['latent'])
        opt_dec.load_state_dict(saved['opt_dec']); opt_lat.load_state_dict(saved['opt_lat'])
        sch_dec.load_state_dict(saved['sch_dec']); sch_lat.load_state_dict(saved['sch_lat'])
        order.set_state(saved['order'].cpu() if hasattr(saved['order'], 'cpu') else saved['order'])
        rows, start_epoch, best = saved['rows'], saved['epoch'] + 1, saved['best']
        log(f'{arm}: resuming DeepSDF training at epoch {start_epoch}')
    log(f'{arm}: DeepSDF on {len(keep)} skulls ({R:,} SDF samples, {nbatches} batches/epoch, {E} epochs, '
        f'beta={beta:g}, matrix={spec["matrix"]}, TF32={TRAIN_TF32})')
    t0 = time.time()
    decoder.train()
    for epoch in range(start_epoch, E + 1):
        perm = torch.randperm(R, device='cuda', generator=order)
        total = torch.zeros(3, device='cuda')
        for nb in range(nbatches):
            sel = perm[nb * B:(nb + 1) * B]
            codes = latent(I[sel])
            pred = decoder(codes, P[sel])
            loss, recon, reg = vd.deepsdf_loss(pred, D[sel].unsqueeze(-1), codes, delta, reg_w)
            if nb % PHYLO_EVERY == 0 and beta > 0:
                loss = loss + beta * phylo(latent.latent_codes.weight)
            opt_dec.zero_grad(set_to_none=True)
            opt_lat.zero_grad(set_to_none=True)
            loss.backward()
            opt_dec.step()
            opt_lat.step()
            total += torch.stack((loss.detach(), recon.detach(), reg.detach()))
        ep_loss, ep_recon, ep_reg = (total / nbatches).tolist()
        sch_dec.step(); sch_lat.step()
        if not np.isfinite(ep_loss):
            raise RuntimeError(f'{arm}: non-finite loss at epoch {epoch}')
        row = {'epoch': epoch, 'loss': ep_loss, 'recon': ep_recon, 'reg': ep_reg}
        if epoch % log_every == 0 or epoch == 1 or epoch == E:
            W = latent.latent_codes.weight.detach().cpu().numpy()
            row['LPC_proxy'] = pair_correlation(W, species, matrix('proxy'))
            row['corr_tree'] = pair_correlation(W, species, matrix('tree'))
            row['LPC_used'] = pair_correlation(W, species, M)
            rate = (time.time() - t0) / (epoch - start_epoch + 1)
            log(f'{arm}: epoch {epoch:4d}/{E}  loss {ep_loss:.6f}  recon {ep_recon:.6f}  LPC(used) {row["LPC_used"]:.3f}  '
                f'LPC(proxy) {row["LPC_proxy"]:.3f}  corr(tree) {row["corr_tree"]:.3f}  '
                f'({rate:.1f} s/epoch, about {rate * (E - epoch) / 3600:.1f} h left)')
        rows.append(row)
        if ep_loss < best['loss']:
            best = {'loss': ep_loss, 'epoch': epoch,
                    'decoder': {k: v.detach().clone() for k, v in decoder.state_dict().items()},
                    'latent': {k: v.detach().clone() for k, v in latent.state_dict().items()}}
        if epoch % 100 == 0 and epoch < E:
            torch.save({'decoder': decoder.state_dict(), 'latent': latent.state_dict(), 'opt_dec': opt_dec.state_dict(),
                        'opt_lat': opt_lat.state_dict(), 'sch_dec': sch_dec.state_dict(), 'sch_lat': sch_lat.state_dict(),
                        'order': order.get_state(), 'rows': rows, 'epoch': epoch, 'best': best,
                        'init_digest': init_digest, 'data_digest': data_digest, 'cfg': cfg}, str(resume) + '.tmp')
            Path(str(resume) + '.tmp').replace(resume)
    save_checkpoint(out / 'final.pt', decoder.state_dict(), latent.state_dict(), E, rows[-1]['loss'], cfg)
    save_checkpoint(out / 'best.pt', best['decoder'], best['latent'], best['epoch'], best['loss'], cfg)
    W = latent.latent_codes.weight.detach().cpu().numpy().astype(np.float32)
    np.savez(out / 'CODES.npz', codes=W, species=species, specimen_index=np.asarray(keep, np.int64),
             specimen=np.asarray([names[i] for i in keep]))
    write_json(out / 'TRAIN_LOG.json', {'arm': arm, 'spec': spec, 'config': cfg, 'specimens': [names[i] for i in keep],
                                        'init_digest': init_digest, 'data_digest': data_digest,
                                        'phylo_loss_equivalence_check': equivalence, 'tf32': TRAIN_TF32,
                                        'samples_per_specimen': SAMPLES_PER_SPECIMEN, 'batches_per_epoch': nbatches,
                                        'updates': nbatches * E, 'best_epoch': best['epoch'], 'best_loss': best['loss'],
                                        'seconds': time.time() - t0, 'epochs': rows})
    write_json(out / 'TRAINING_COMPLETE.json', {'arm': arm, 'final_sha256': sha(out / 'final.pt'),
                                                'best_sha256': sha(out / 'best.pt'), 'epochs': E})
    if resume.exists():
        resume.unlink()
    del P, D, I
    torch.cuda.empty_cache()
    log(f'{arm}: DeepSDF finished in {(time.time() - t0) / 3600:.2f} h (final loss {rows[-1]["loss"]:.6f}, '
        f'best epoch {best["epoch"]})')
    return out


def load_codes(arm, out=None):
    out = Path(out) if out is not None else deepsdf_dir(arm)
    with np.load(out / 'CODES.npz', allow_pickle=False) as data:
        return (data['codes'].astype(np.float32), data['species'].astype(np.int64),
                data['specimen_index'].astype(np.int64), [str(x) for x in data['specimen']])


# ---------------------------------------------------------------- decoding (256^3) -----------
_dec = {}


def decoder_for(ckpt):
    """Any checkpoint in the paper's format, loaded with the frozen restricted loader and decoded exactly as in
    the frozen benchmark (FP32, TF32 off, deterministic); NumPy/torch parity is checked."""
    import torch
    key = str(Path(ckpt).resolve())
    if key not in _dec:
        exact_mode()
        frozen = MESHSRC / 'grounded' / 'probe_source' / 'frozen'
        backend = load_module(frozen / 'runtime_backend.py', 'rep_frozen_backend')
        bench = load_module(frozen / 'benchmark.py', 'rep_frozen_benchmark')
        source = load_module(frozen / 'deepsdf.py', 'rep_original_deepsdf')
        ck = backend.load_checkpoint(ckpt)
        cfg = ck['config']
        keys = ['latent_dim', 'hidden_dim', 'num_layers', 'skip_connection', 'weight_norm']
        model = source.DeepSDFDecoder(**{k: cfg[k] for k in keys if k in cfg})
        model.load_state_dict({k: torch.from_numpy(np.array(v, copy=True)) for k, v in ck['decoder_state_dict'].items()})
        model.to('cuda').eval()
        points = np.random.default_rng(20261014).uniform(-1, 1, (127, 3)).astype(np.float32)
        z = ck['latent_lib_state_dict']['latent_codes.weight'][0]
        with torch.inference_mode():
            values = model(torch.as_tensor(z, dtype=torch.float32, device='cuda').expand(127, -1),
                           torch.as_tensor(points, device='cuda')).reshape(-1).cpu().numpy()
        parity = float(np.max(np.abs(values - backend.NumpyDecoder(ck).predict(points, z))))
        if parity > 3e-5:
            raise RuntimeError(f'NumPy/torch decoder parity failed: {parity}')
        _dec.clear()
        _dec[key] = (model, bench, parity, sha(ckpt))
    return _dec[key]


def sdf_field(model, code, resolution=RES, chunk=8192):
    import torch
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


def decode_rows(ckpt, rows, root, paired_reference, training_seed, resolution=RES):
    """rows: dicts with 'task', 'code', 'method' and either ('cohort', 'sample_slot', 'species') for generated
    skulls or ('specimen', 'species') with method 'reference_reconstruction'. Failures stay in the records."""
    root = Path(root)
    if (root / 'summary.json').exists():
        return root
    if BUDGET['rows_per_method']:
        kept, seen = [], {}
        for row in rows:
            key = (row['method'], row.get('role'))
            seen[key] = seen.get(key, 0) + 1
            if seen[key] <= BUDGET['rows_per_method']:
                kept.append(row)
        rows = kept
    model, bench, parity, ck_sha = decoder_for(ckpt)
    exact_mode()
    if root.exists():
        shutil.rmtree(root)
    (root / 'tasks').mkdir(parents=True)
    records, start = [], time.time()
    for done, row in enumerate(rows, 1):
        task = row['task']
        folder = root / 'tasks' / task
        folder.mkdir()
        code = np.asarray(row['code'], dtype=np.float32)
        record = {k: v for k, v in row.items() if k != 'code'}
        record.update(group=STUDY_GROUP, historical_arm=paired_reference, paired_reference=paired_reference,
                      training_seed=training_seed, generation_seed=training_seed, decoder=Path(ckpt).name,
                      decoder_sha256=ck_sha, resolution=resolution, zero_level=0, spacing=2 / (resolution - 1),
                      component_filtering='none', grid_frame='decoder normalized [-1,1]^3',
                      finite_input_code=bool(np.isfinite(code).all()))
        t0 = time.time()
        try:
            if not record['finite_input_code']:
                raise ValueError('nonfinite latent code')
            field = sdf_field(model, code, resolution)
            record.update(sdf_min=float(field.min()), sdf_max=float(field.max()))
            vertices, faces, extractor = bench.extract(field, 'skimage')
            record.update(bench.mesh_info(vertices, faces))
            np.savez(folder / 'mesh.npz', vertices=vertices, faces=faces)
            record.update(status='completed', extractor=extractor, mesh_path=f'tasks/{task}/mesh.npz',
                          output_sha256={'mesh.npz': sha(folder / 'mesh.npz')})
        except Exception as error:      # every failure stays in the denominator
            record.update(status='mesh_failed', failure_type=type(error).__name__, failure=str(error))
        record['task_wall_seconds'] = time.time() - t0
        records.append(record)
        if done % 20 == 0 or done == len(rows):
            log(f'  decoded {done}/{len(rows)} meshes ({(time.time() - start) / done:.1f} s/mesh)')
    write_json(root / 'run_config.json', {'tasks': [{'task': r['task']} for r in records],
                                          'paired_reference': paired_reference, 'training_seed': training_seed,
                                          'decoder_sha256': ck_sha, 'decoder_parity_max_abs': parity})
    write_json(root / 'summary.json', {'schema': 'rep_decode_v1', 'records': records, 'resolution': resolution,
                                       'valid_meshes': sum(r['status'] == 'completed' for r in records)})
    return root


# ---------------------------------------------------------------- frozen metrics -------------
_vendor = None


def vendor():
    """Frozen vendor_metrics.py, unchanged bytes; the accepted method-name registry is extended, references are
    cached, and the frozen evaluator's exact KD-tree Chamfer distance queries with all CPU cores (values unchanged)."""
    global _vendor
    if _vendor is None:
        vm = load_module(MESHSRC / 'vendor' / 'vendor_metrics.py', 'rep_vendor_metrics')
        vm.METHODS = tuple(vm.METHODS) + METHOD_NAMES
        original, cache = vm.load_references, {}

        def load_references(manifest):
            key = str(Path(manifest).resolve())
            if key not in cache:
                cache[key] = original(manifest)
            return cache[key]

        vm.load_references, vm.cd = load_references, kd_cd
        _vendor = vm
    return _vendor


def kd_cd(left, right, left_tree=None, right_tree=None):
    """The frozen evaluator's exact KD-tree Chamfer distance; only the query uses all CPU cores."""
    from scipy.spatial import cKDTree
    left_tree = cKDTree(left) if left_tree is None else left_tree
    right_tree = cKDTree(right) if right_tree is None else right_tree
    value = float(np.square(left_tree.query(right, workers=-1)[0]).mean()
                  + np.square(right_tree.query(left, workers=-1)[0]).mean())
    if not np.isfinite(value) or value < 0:
        raise ValueError('Invalid Chamfer value')
    return value


def _nearest(query, ref, block=2048):
    import torch
    parts = []
    for i in range(0, len(query), block):
        q = query[i:i + block]
        try:
            d = torch.cdist(q, ref, compute_mode='donot_use_mm_for_euclid_dist')
        except RuntimeError:
            d = (q[:, None, :] - ref[None, :, :]).pow(2).sum(-1).sqrt()
        parts.append(d.min(1).values)
    return torch.cat(parts)


def gpu_cd(left, right, left_tree=None, right_tree=None):
    """Same quantity as the frozen KD-tree Chamfer distance (exact nearest neighbours), float64 on the GPU."""
    import torch
    a = torch.from_numpy(np.ascontiguousarray(left, dtype=np.float64)).to('cuda')
    b = torch.from_numpy(np.ascontiguousarray(right, dtype=np.float64)).to('cuda')
    value = float((_nearest(b, a).square().mean() + _nearest(a, b).square().mean()).item())
    if not np.isfinite(value) or value < 0:
        raise ValueError('Invalid Chamfer value')
    return value


def evaluate_dir(decode_root, out, keep_meshes=False):
    out = Path(out)
    if (out / 'summary.json').exists():
        return json.loads((out / 'summary.json').read_text())
    if out.exists():
        shutil.rmtree(out)
    start = time.time()
    result = vendor().run(REFS / 'reference_manifest.json', [decode_root], out, points=10000,
                          surface_seeds=SURFACE_SEEDS, resume=False)
    shutil.rmtree(out / 'surface_samples', ignore_errors=True)     # large, regenerable
    if not keep_meshes:
        shutil.rmtree(Path(decode_root) / 'tasks', ignore_errors=True)
    log(f'  metrics done in {time.time() - start:.0f} s')
    return result


def reconstruction_table(metrics_out):
    rows = json.loads((Path(metrics_out) / 'reconstruction_per_specimen.json').read_text())
    return {r['specimen']: r.get('mean_matched_specimen_CD') for r in rows}


def group_brief(result, title):
    lines = []
    for g in sorted(result.get('groups', []), key=lambda g: (g['method'], g['cohort'])):
        s = g.get('geometry_surface_seed_summary', {})
        get = lambda k: s.get(k, {}).get('mean', float('nan'))
        f = g.get('morphfid_decoder_frame_primary', {})
        f8 = f.get('geometry8_no_count_or_centroid', {}).get('value', float('nan')) if isinstance(f, dict) else float('nan')
        lines.append(f"    {g['method']:24s} {g['cohort'].replace('unconditional_species_mixture', 'mixture'):26s} "
                     f"valid {g['valid_meshes']:3d}/{g['attempts']:3d}  NN-CD {get('generated_to_reference_nn_CD_mean') * 1e3:5.2f}  "
                     f"MMD {get('MMD_CD_reference_to_generated') * 1e3:5.2f}  COV {get('COV_reference_fraction'):4.2f}  "
                     f"div {get('diversity_all_pairs_CD_mean') * 1e3:5.2f}  MorphFID8 {f8:5.2f}")
    log(title + ' (CDs x1e-3)\n' + '\n'.join(lines))


# ---------------------------------------------------------------- pooled residual flow --------
_rmc = None


def rmc():
    """The original run_matched_cfm module (used for its frozen network, seeding and determinism helpers)."""
    global _rmc
    if _rmc is None:
        _rmc = load_module(SRC / 'run_matched_cfm.py', 'rep_run_matched_cfm')
    return _rmc


def flow_stats(codes, species, num_species=24):
    """run_matched_cfm.normalize_real restricted to the species present: centroids from the real codes, residual
    mean vector and scalar unbiased residual standard deviation."""
    import torch
    codes_t = torch.as_tensor(np.asarray(codes, np.float32))
    species_t = torch.as_tensor(np.asarray(species, np.int64))
    centroids = torch.zeros((num_species, codes_t.shape[1]), dtype=torch.float32)
    for i in sorted(set(np.asarray(species).tolist())):
        centroids[i] = codes_t[species_t == i].mean(0)
    residuals = codes_t - centroids[species_t]
    return {'centroids': centroids.numpy(), 'residual_mean': residuals.mean(0).numpy(),
            'residual_global_std': np.asarray(residuals.std().item() + 1e-8, dtype=np.float32)}


def train_flow(codes, species, seed, updates=10000):
    """The final model's residual flow, exactly as the matched protocol trains its pooled (neutral) arm:
    PhyloFlowNet (3 blocks x 128), null species token and zero size/phylogeny inputs, AdamW 1e-4 (wd 1e-5), cosine
    schedule, batch 256 with replacement, gradient clipping 1, variance term 0.1 on the one-step midpoint estimate."""
    import torch
    m = rmc()
    m.deterministic()
    stats = flow_stats(codes, species)
    codes = np.asarray(codes, np.float32)
    raw = codes - stats['centroids'][np.asarray(species)]
    target = (raw - stats['residual_mean']) / stats['residual_global_std']
    n = len(codes)
    m.seed_all(seed)
    net = m.load_vendor('diffusion').PhyloFlowNet(**m.model_kwargs(32))
    initial = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    m.seed_all(seed)
    model = m.load_vendor('diffusion').PhyloFlowNet(**m.model_kwargs(32)).to('cuda')
    model.load_state_dict(initial, strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0001, weight_decay=.00001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=updates, eta_min=.000001)
    tensors = {'target': torch.as_tensor(target), 'species': torch.full((n,), 24, dtype=torch.long),
               'bbox': torch.zeros((n, 3), dtype=torch.float32), 'phylo': torch.zeros((n, 32), dtype=torch.float32)}
    weights = np.ones(n, dtype=np.float64)
    weights[:100] = 3.
    weights = torch.as_tensor(weights)
    sampler = torch.Generator(device='cpu').manual_seed(seed + 200000)
    flow = torch.Generator(device='cuda').manual_seed(seed + 300000)
    m.seed_all(seed + 400000)
    trace = []
    model.train()
    for step in range(1, updates + 1):
        indices = torch.multinomial(weights, 256, replacement=True, generator=sampler)
        batch = {k: value[indices].to('cuda') for k, value in tensors.items()}
        x1, sp, bbox, phy = batch['target'], batch['species'], batch['bbox'], batch['phylo']
        dropped = bool(torch.rand((), device='cuda', generator=flow).item() < .1)
        if dropped:
            sp = torch.full_like(sp, 24)
        t = torch.rand((256,), device='cuda', generator=flow)
        x0 = torch.randn(x1.shape, device='cuda', dtype=x1.dtype, generator=flow)
        xt = (1 - t[:, None]) * x0 + t[:, None] * x1
        predicted = model(xt, t, sp, bbox, phy)
        flow_loss = torch.nn.functional.mse_loss(predicted, x1 - x0)
        mid = .5 * x0 + .5 * x1
        mid_velocity = model(mid, torch.full((256,), .5, device='cuda'), sp, bbox, phy)
        estimate = mid + .5 * mid_velocity
        variance_loss = torch.relu(x1.var() - estimate.var())
        loss = flow_loss + .1 * variance_loss
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('non-finite flow loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step(); scheduler.step()
        if step % 100 == 0 or step == updates:
            trace.append({'update': step, 'loss': float(loss.detach().cpu()), 'flow_loss': float(flow_loss.detach().cpu()),
                          'variance_loss': float(variance_loss.detach().cpu())})
    model.eval()
    return model, stats, trace


def generation_inputs(seed):
    """The matched protocol's frozen generation schedule: 4 focal species x 10, then 100 mixture slots
    (species = slot mod 24); initial noise from torch.Generator(cpu) seeded with seed + 100000."""
    import torch
    names = species_names()
    species, cohort, slot = [], [], []
    for name in FOCAL:
        for k in range(10):
            species.append(names.index(name)); cohort.append('conditional_' + name); slot.append(k)
    for k in range(100):
        species.append(k % 24); cohort.append('unconditional_species_mixture'); slot.append(k)
    noise = torch.randn((140, 64), generator=torch.Generator(device='cpu').manual_seed(seed + 100000)).numpy()
    return np.asarray(species, np.int64), np.asarray(cohort), np.asarray(slot, np.int64), noise


def flow_sample(model, noise, temperature=1.0, steps=50):
    """50 Euler steps, guidance 1, neutral conditioning (identical arithmetic to the final-model sampler)."""
    import torch
    x = torch.as_tensor(np.asarray(noise, np.float32), device='cuda').clone() * float(temperature)
    n = len(x)
    species = torch.full((n,), 24, device='cuda', dtype=torch.long)
    bbox = torch.zeros((n, 3), device='cuda')
    phylo = torch.zeros((n, 32), device='cuda')
    with torch.no_grad():
        for step in range(steps):
            t = torch.full((n,), step / float(steps), device='cuda')
            x += model(x, t, species, bbox, phylo) / float(steps)
    return x.cpu().numpy()


def residuals_from_targets(x, stats):
    return x * stats['residual_global_std'] + stats['residual_mean']


# ---------------------------------------------------------------- study: loss ablation --------
def arm_spec(arm):
    for study in ('loss_ablation', 'shuffle_control'):
        if arm in STUDIES[study]:
            return STUDIES[study][arm]
    raise ValueError(arm)


def run_loss_ablation_arm(arm):
    spec = arm_spec(arm)
    t_arm = time.time()
    ckdir = train_deepsdf(arm, spec, epochs=BUDGET['epochs'])
    ckpt = ckdir / 'final.pt'
    codes, species, keep, names = load_codes(arm)
    res = OUT / arm
    if not (res / 'LATENT_SUMMARY.json').exists():
        summary = latent_summary(codes, species, used=spec['matrix'])
        write_json(res / 'LATENT_SUMMARY.json', summary)
        log(f'{arm}: latent geometry ' + json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in summary.items()}))
        path = package(f'shuffle_{arm}_latent_only')
        log(f'{arm}: key result saved early in {path} (latent geometry and the trained decoder)')
    # reconstruction of all 100 training skulls
    rec_out = res / 'metrics_reconstruction'
    if not (rec_out / 'summary.json').exists():
        sp_names = species_names()
        rows = [{'task': f'{arm}__recon__{k:03d}__{names[k]}', 'code': codes[k], 'method': 'reference_reconstruction',
                 'specimen': names[k], 'species': sp_names[species[k]], 'species_idx': int(species[k]),
                 'cohort': 'reconstruction', 'sample_slot': k, 'role': 'train_learned'} for k in range(len(codes))]
        log(f'{arm}: decoding the {len(rows)} training skulls at {RES}^3')
        root = decode_rows(ckpt, rows, WORK / arm / 'decode_reconstruction', arm, 0)
        evaluate_dir(root, rec_out)
    table = reconstruction_table(rec_out)
    vals = np.array([v for v in table.values() if v is not None])
    if len(vals):
        log(f'{arm}: reconstruction CD mean {vals.mean() * 1e3:.3f}e-3, median {np.median(vals) * 1e3:.3f}e-3, '
            f'max {vals.max() * 1e3:.3f}e-3, valid {len(vals)}/{len(table)}')
    else:
        log(f'{arm}: no valid reconstruction meshes ({len(table)} attempts)')
    # generation with the final model's pooled residual flow
    for seed in GEN_SEEDS:
        gen_dir = res / f'generation_seed{seed}'
        codes_path = gen_dir / 'GENERATED_CODES.npz'
        if not codes_path.exists():
            t0 = time.time()
            model, stats, trace = train_flow(codes, species, seed, updates=BUDGET['flow_updates'])
            sp_gen, cohort, slot, noise = generation_inputs(seed)
            x = flow_sample(model, noise)
            generated = (residuals_from_targets(x, stats) + stats['centroids'][sp_gen]).astype(np.float32)
            if not np.isfinite(generated).all():
                raise RuntimeError('non-finite generated codes')
            gen_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(codes_path, codes=generated, normalized_targets=x, species=sp_gen, cohort=cohort,
                                sample_slot=slot)
            residual_var_ratio = float(np.var(x, ddof=1)) / 1.0
            write_json(gen_dir / 'FLOW.json', {'seed': seed, 'trace': trace, 'seconds': time.time() - t0,
                                               'generated_normalized_residual_variance': residual_var_ratio,
                                               'stats': {k: v for k, v in stats.items() if k != 'centroids'}})
            log(f'{arm}: residual flow seed {seed} trained in {time.time() - t0:.0f} s; generated/training residual '
                f'variance {residual_var_ratio:.3f}')
        data = np.load(codes_path, allow_pickle=False)
        sp_names = species_names()
        rows = [{'task': f'{arm}__gen__seed{seed}__{str(data["cohort"][r])}__slot{int(data["sample_slot"][r]):03d}',
                 'code': data['codes'][r], 'method': 'rep_pooled_residual_cfm', 'cohort': str(data['cohort'][r]),
                 'sample_slot': int(data['sample_slot'][r]), 'species': sp_names[int(data['species'][r])],
                 'species_idx': int(data['species'][r]), 'source_array_row': r} for r in range(len(data['codes']))]
        met_out = res / f'metrics_generation_seed{seed}'
        if not (met_out / 'summary.json').exists():
            log(f'{arm}: decoding 140 generated skulls (seed {seed})')
            root = decode_rows(ckpt, rows, WORK / arm / f'decode_generation_seed{seed}', arm, seed)
            evaluate_dir(root, met_out)
        group_brief(json.loads((met_out / 'summary.json').read_text()), f'{arm} generation seed {seed}')
    write_json(res / 'ARM_COMPLETE.json', {'arm': arm, 'spec': spec, 'hours': (time.time() - t_arm) / 3600})


# ---------------------------------------------------------------- study: held-out species -----
def fit_codes(ckpt, specimen_indices, init_codes, seed, steps=None, label='fit'):
    """Fit one latent code per skull with the decoder frozen: the training objective (clamped L1 + 1e-3 code
    penalty), Adam lr 1e-3, 4,096 SDF samples per skull per step, start at the training-code mean + 0.01 jitter."""
    import torch
    steps = int(steps or FIT['steps'])
    fp32_mode()
    vd = vendor_deepsdf()
    ck = torch.load(ckpt, map_location='cpu', weights_only=True)
    decoder = vd.DeepSDFDecoder(latent_dim=64, hidden_dim=512, num_layers=8, skip_connection=[4], weight_norm=True)
    decoder.load_state_dict(ck['decoder_state_dict'])
    decoder.to('cuda').eval()
    for p in decoder.parameters():
        p.requires_grad_(False)
    names, _ = specimens()
    pts, sdf = [], []
    for i in specimen_indices:
        with np.load(processed_path(names[i]), allow_pickle=False) as data:
            pts.append(np.asarray(data['points'], np.float32)); sdf.append(np.asarray(data['sdf'], np.float32).reshape(-1))
    n_points = min(len(p) for p in pts)
    Pt = torch.as_tensor(np.stack([p[:n_points] for p in pts]), device='cuda')      # (S, n, 3)
    Dt = torch.as_tensor(np.stack([d[:n_points] for d in sdf]), device='cuda')      # (S, n)
    S, K = len(specimen_indices), FIT['points_per_specimen']
    gen = torch.Generator(device='cuda').manual_seed(seed + 600000)
    mean = torch.as_tensor(np.asarray(init_codes, np.float32).mean(0), device='cuda')
    z = (mean[None, :] + FIT['init_jitter'] * torch.randn((S, 64), device='cuda', generator=gen)).requires_grad_(True)
    opt = torch.optim.Adam([z], lr=FIT['lr'])
    rows = torch.arange(S, device='cuda')[:, None].expand(S, K)
    trace = []
    for step in range(1, steps + 1):
        idx = torch.randint(0, n_points, (S, K), device='cuda', generator=gen)
        p = Pt[rows, idx].reshape(-1, 3)
        d = Dt[rows, idx].reshape(-1, 1)
        codes = z.repeat_interleave(K, dim=0)          # row order matches rows.reshape(-1)
        loss, recon, reg = vd.deepsdf_loss(decoder(codes, p), d, codes, DEEPSDF['clamping_distance'],
                                           DEEPSDF['latent_reg_weight'])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 100 == 0 or step == steps:
            trace.append({'step': step, 'loss': float(loss), 'recon': float(recon)})
    log(f'  {label}: fitted {S} codes ({steps} steps), final recon {trace[-1]["recon"]:.6f}')
    return z.detach().cpu().numpy().astype(np.float32), trace


def predict_centroids(train_codes, train_species, heldout_species, names_sp):
    """Centroid of an unseen species from the training species: inverse-distance weighted mean of the 3 nearest
    training species (proxy or dated tree; ties broken alphabetically), or the mean of all training centroids."""
    proxy, tree = matrix('proxy'), matrix('tree')
    present = sorted(set(train_species.tolist()))
    C = {s: train_codes[train_species == s].mean(0) for s in present}
    out = {}
    for h in heldout_species:
        entry = {}
        for label, M in (('proxy_knn', proxy), ('tree_knn', tree)):
            near = sorted(present, key=lambda s: (M[h, s], names_sp[s]))[:3]
            w = np.array([1.0 / (M[h, s] + 1e-6) for s in near])
            entry[label] = {'centroid': (sum(wi * C[s] for wi, s in zip(w, near)) / w.sum()).astype(np.float32),
                            'neighbours': [names_sp[s] for s in near], 'distances': [float(M[h, s]) for s in near],
                            'weights': (w / w.sum()).tolist()}
        entry['global_mean'] = {'centroid': np.mean([C[s] for s in present], axis=0).astype(np.float32),
                                'neighbours': 'all training species, equal weight'}
        out[h] = entry
    return out


def real_real_cd(rows_a, rows_b, points=10000):
    """Chamfer distances between real skulls (frozen reference frame and surface sampling, both seeds averaged)."""
    vm = vendor()
    refs, _ = vm.load_references(REFS / 'reference_manifest.json')
    by_name = {r['specimen']: r for r in refs}
    cache = {}

    def cloud(name):
        if name not in cache:
            r = by_name[name]
            cache[name] = [vm.sample_surface(r['vertices'], r['faces'], points, s) for s in SURFACE_SEEDS]
        return cache[name]
    D = np.zeros((len(rows_a), len(rows_b)))
    for i, a in enumerate(rows_a):
        for j, b in enumerate(rows_b):
            D[i, j] = np.mean([kd_cd(x, y) for x, y in zip(cloud(a), cloud(b))])
    return D


def run_heldout_arm(arm):
    spec = STUDIES['heldout'][arm]
    t_arm = time.time()
    res = OUT / arm
    ckdir = train_deepsdf(arm, spec, epochs=BUDGET['epochs'])
    ckpt = ckdir / 'final.pt'
    codes, species, keep, names_train = load_codes(arm)
    sp_names = species_names()
    all_names, all_species = specimens()
    if not (res / 'LATENT_SUMMARY.json').exists():
        write_json(res / 'LATENT_SUMMARY.json', latent_summary(codes, species))
    held = selection('heldout21')
    fit_path = res / 'FITTED_CODES.npz'
    if not fit_path.exists():
        # control: refit 5 training skulls (one from each of the 5 largest training species) with the same procedure
        counts = {s: int((species == s).sum()) for s in sorted(set(species.tolist()))}
        top = sorted(counts, key=lambda s: (-counts[s], sp_names[s]))[:5]
        controls = [int(keep[np.flatnonzero(species == s)[0]]) for s in top]
        z_held, trace_held = fit_codes(ckpt, held, codes, SEED, steps=BUDGET['fit_steps'], label='withheld skulls')
        z_ctrl, trace_ctrl = fit_codes(ckpt, controls, codes, SEED + 1, steps=BUDGET['fit_steps'],
                                       label='training-skull controls')
        res.mkdir(parents=True, exist_ok=True)
        np.savez(fit_path, heldout_codes=z_held, heldout_index=np.asarray(held), control_codes=z_ctrl,
                 control_index=np.asarray(controls))
        write_json(res / 'FIT_LOG.json', {'fit': FIT, 'heldout_trace': trace_held, 'control_trace': trace_ctrl,
                                          'heldout': [all_names[i] for i in held], 'controls': [all_names[i] for i in controls]})
    fit = np.load(fit_path, allow_pickle=False)
    z_held, held = fit['heldout_codes'], [int(i) for i in fit['heldout_index']]
    z_ctrl, controls = fit['control_codes'], [int(i) for i in fit['control_index']]
    held_species = np.asarray([all_species[i] for i in held])
    focal_idx = [sp_names.index(n) for n in FOCAL]
    preds = predict_centroids(codes, species, focal_idx, sp_names)
    for h in focal_idx:
        preds[h]['oracle'] = {'centroid': z_held[held_species == h].mean(0).astype(np.float32),
                              'neighbours': 'mean of the fitted codes of the withheld skulls (reference only)'}
    errors = {sp_names[h]: {p: float(np.linalg.norm(preds[h][p]['centroid'] - preds[h]['oracle']['centroid']))
                            for p in PREDICTORS} for h in focal_idx}
    write_json(res / 'CENTROID_PREDICTIONS.json', {
        'latent_error_vs_fitted_centroid': errors,
        'predictions': {sp_names[h]: {p: {k: v for k, v in preds[h][p].items()} for p in PREDICTORS} for h in focal_idx}})
    log(f'{arm}: centroid error vs fitted centroid ' + json.dumps({k: {p: round(v, 3) for p, v in e.items()} for k, e in errors.items()}))
    # residuals for the unseen species: pooled flow trained on the 79 training codes only, frozen noise rows 0-39
    seed = GEN_SEEDS[0]
    gen_path = res / f'HELDOUT_GENERATED_CODES_seed{seed}.npz'
    if not gen_path.exists():
        t0 = time.time()
        model, stats, trace = train_flow(codes, species, seed, updates=BUDGET['flow_updates'])
        sp_gen, cohort, slot, noise = generation_inputs(seed)
        x = flow_sample(model, noise[:40])
        r_hat = residuals_from_targets(x, stats).astype(np.float32)
        arrays = {'residuals': r_hat, 'species': sp_gen[:40], 'cohort': cohort[:40], 'sample_slot': slot[:40]}
        for p in PREDICTORS:
            arrays['codes_' + p] = np.stack([preds[int(s)][p]['centroid'] for s in sp_gen[:40]]) + r_hat
        np.savez_compressed(gen_path, **arrays)
        write_json(res / f'FLOW_seed{seed}.json', {'seed': seed, 'trace': trace, 'seconds': time.time() - t0,
                                                   'generated_normalized_residual_variance': float(np.var(x, ddof=1))})
    gen = np.load(gen_path, allow_pickle=False)
    met_out = res / 'metrics'
    if not (met_out / 'summary.json').exists():
        rows = []
        for k, i in enumerate(held):
            rows.append({'task': f'{arm}__heldout_fit__{all_names[i]}', 'code': z_held[k], 'method': 'reference_reconstruction',
                         'specimen': all_names[i], 'species': sp_names[all_species[i]], 'species_idx': int(all_species[i]),
                         'cohort': 'reconstruction', 'sample_slot': k, 'role': 'heldout_fit'})
        for k, i in enumerate(controls):
            learned = codes[list(keep).index(i)]
            rows.append({'task': f'{arm}__control_fit__{all_names[i]}', 'code': z_ctrl[k], 'method': 'reference_reconstruction',
                         'specimen': all_names[i], 'species': sp_names[all_species[i]], 'species_idx': int(all_species[i]),
                         'cohort': 'reconstruction', 'sample_slot': 100 + k, 'role': 'train_fit_control'})
            rows.append({'task': f'{arm}__control_learned__{all_names[i]}', 'code': learned, 'method': 'reference_reconstruction',
                         'specimen': all_names[i], 'species': sp_names[all_species[i]], 'species_idx': int(all_species[i]),
                         'cohort': 'reconstruction', 'sample_slot': 200 + k, 'role': 'train_learned'})
        for h in focal_idx:
            for p in PREDICTORS:
                rows.append({'task': f'{arm}__centroid__{p}__{sp_names[h]}', 'code': preds[h][p]['centroid'],
                             'method': 'ho_centroid_' + p, 'cohort': 'conditional_' + sp_names[h], 'sample_slot': 0,
                             'species': sp_names[h], 'species_idx': int(h)})
        for p in PREDICTORS:
            for r in range(40):
                rows.append({'task': f'{arm}__gen__{p}__{str(gen["cohort"][r])}__slot{int(gen["sample_slot"][r]):03d}',
                             'code': gen['codes_' + p][r], 'method': 'ho_gen_' + p, 'cohort': str(gen['cohort'][r]),
                             'sample_slot': int(gen['sample_slot'][r]), 'species': sp_names[int(gen['species'][r])],
                             'species_idx': int(gen['species'][r]), 'source_array_row': r})
        log(f'{arm}: decoding {len(rows)} meshes (21 withheld fits, 10 controls, 16 centroids, 160 generated)')
        root = decode_rows(ckpt, rows, WORK / arm / 'decode_heldout', arm, seed)
        evaluate_dir(root, met_out)
    result = json.loads((met_out / 'summary.json').read_text())
    group_brief(result, f'{arm} withheld species')
    rec = json.loads((met_out / 'reconstruction_per_specimen.json').read_text())
    by_role = {}
    for r in rec:
        if r.get('mean_matched_specimen_CD') is not None:
            by_role.setdefault(r.get('role'), []).append(r['mean_matched_specimen_CD'])
    log(f'{arm}: reconstruction CD x1e-3 by role ' + json.dumps({k: round(float(np.mean(v)) * 1e3, 3) for k, v in by_role.items()}))
    write_json(res / 'ARM_COMPLETE.json', {'arm': arm, 'spec': spec, 'hours': (time.time() - t_arm) / 3600})


def real_skull_distances():
    """CD from each withheld real skull to every real skull (for retrieval baselines and real-to-real novelty)."""
    path = OUT / 'REAL_WITHHELD_TO_ALL_CD.npz'
    if path.exists():
        return path
    names, _ = specimens()
    held = [names[i] for i in selection('heldout21')]
    t0 = time.time()
    D = real_real_cd(held, names)
    np.savez(path, cd=D, rows=np.asarray(held), cols=np.asarray(names))
    log(f'real-skull distances ({len(held)} x {len(names)}) in {time.time() - t0:.0f} s')
    return path


# ---------------------------------------------------------------- validation -----------------
def validate(flow_check=True):
    """Checks before the long run (about 10-15 minutes):
    1. the generalised decoder + frozen metrics reproduce the paper decoder's reconstruction CD of 2 skulls;
    2. the vectorised phylogenetic loss equals the original loop (value and gradient), on the paper codes;
    3. the standalone residual-flow trainer equals the matched protocol's pooled arm (100 updates, identical
       weights expected);
    4. a 2-epoch DeepSDF smoke run on 4 skulls produces a checkpoint that the frozen loader decodes."""
    import torch
    report = {}
    vdir = OUT / 'validation'
    vdir.mkdir(parents=True, exist_ok=True)
    # 1
    paper = SRC / 'inputs' / 'final.pt'
    ck = torch.load(paper, map_location='cpu', weights_only=True)
    W = ck['latent_lib_state_dict']['latent_codes.weight'].numpy()
    names, species = specimens()
    rows = [{'task': f'check__recon__{k:03d}__{names[k]}', 'code': W[k], 'method': 'reference_reconstruction',
             'specimen': names[k], 'species': species_names()[species[k]], 'species_idx': int(species[k]),
             'cohort': 'reconstruction', 'sample_slot': k, 'role': 'paper_decoder'} for k in (0, 1)]
    root = decode_rows(paper, rows, vdir / 'decode_check', 'paper_final', 0)
    evaluate_dir(root, vdir / 'metrics_check')
    table = reconstruction_table(vdir / 'metrics_check')
    checks = {n: {'expected': FROZEN_RECON[n], 'obtained': table[n],
                  'relative_difference': abs(table[n] - FROZEN_RECON[n]) / FROZEN_RECON[n]} for n in FROZEN_RECON}
    worst = max(c['relative_difference'] for c in checks.values())
    report['decode_metric_reproduction'] = checks
    log(f'check 1 (paper decoder reconstruction, 2 skulls): max relative difference {worst:.2e}')
    if worst > 1e-3:
        write_json(vdir / 'VALIDATION.json', report)
        raise RuntimeError('Decode/metric reproduction failed. Stop and send VALIDATION.json to Claude.')
    # 2
    training_mode()
    Wt = torch.as_tensor(W, device='cuda')
    report['shuffles'] = check_shuffles()
    for kind in ('proxy', 'tree', 'proxy_shuf1'):
        report[f'phylo_loss_equivalence_{kind}'] = _check_phylo_equivalence(
            Wt, species, matrix(kind), _phylo_loss_fn(len(W), species, matrix(kind)))
    report['paper_codes_LPC_proxy'] = pair_correlation(W, species, matrix('proxy'))
    log(f'check 2 (vectorised phylogenetic loss = original loop): passed; paper LPC {report["paper_codes_LPC_proxy"]:.4f}')
    # 3
    if flow_check:
        m = rmc()
        variant = {'augmentation': False, 'phylo_conditioning': False, 'target': 'residual', 'var_weight': .1, 'neutral': True}
        m.VARIANTS['rep_pooled_residual'] = variant
        original_tensors = m.training_tensors

        def training_tensors(arrays, name):
            tensors, weights = original_tensors(arrays, name)
            if m.VARIANTS[name].get('neutral'):
                tensors['species'] = torch.full_like(tensors['species'], 24)
                tensors['bbox'] = torch.zeros_like(tensors['bbox'])
                tensors['phylo'] = torch.zeros_like(tensors['phylo'])
            return tensors, weights
        m.training_tensors = training_tensors
        prepared = m.prepare(base=str(SRC), work=str(vdir / 'rmc_work'), pair='final', seed=42, device='cuda')
        out = vdir / 'rmc_pooled_100'
        if out.exists():
            shutil.rmtree(out)
        m.train(str(prepared), str(out), variant='rep_pooled_residual', updates=100, device='cuda', checkpoint_every=100)
        latest = json.loads((out / 'LATEST.json').read_text())
        reference = m.verified_torch_load(out / latest['checkpoint'])['model']
        mine, _, _ = train_flow(W, species, 42, updates=100)
        diff = max(float((mine.state_dict()[k].cpu() - reference[k].cpu()).abs().max()) for k in reference)
        report['flow_trainer_vs_matched_protocol_max_abs_param_diff'] = diff
        log(f'check 3 (standalone flow trainer vs matched protocol, 100 updates): max |param diff| {diff:.2e}')
        if diff > 1e-5:
            write_json(vdir / 'VALIDATION.json', report)
            raise RuntimeError('Flow trainer differs from the matched protocol. Stop and send VALIDATION.json to Claude.')
    # 4
    smoke = vdir / 'deepsdf_smoke'
    if smoke.exists():
        shutil.rmtree(smoke)
    keep = [int(np.flatnonzero(species == s)[0]) for s in (0, 7, 15, 21)]
    train_deepsdf('smoke', {'beta': 5e-3, 'matrix': 'proxy', 'specimens': 'all'}, epochs=2, keep=keep, out=smoke,
                  log_every=1)
    model, _, parity, _ = decoder_for(smoke / 'final.pt')
    report['smoke_checkpoint_parity'] = parity
    z, _ = fit_codes(smoke / 'final.pt', [2, 3], load_codes('smoke', smoke)[0], SEED, steps=20, label='smoke fit')
    report['smoke_fit_finite'] = bool(np.isfinite(z).all())
    log(f'check 4 (DeepSDF smoke run, frozen loader parity {parity:.1e}, latent fitting ok)')
    shutil.rmtree(smoke, ignore_errors=True)
    write_json(vdir / 'VALIDATION.json', report)
    log('all checks passed')
    return report


def dry_run(study, arm=None):
    """Every step of one arm with tiny budgets (2 epochs, 100 flow updates, 20 fitting steps, 2 meshes per method),
    in a separate folder that is deleted afterwards. Catches errors in minutes instead of after hours."""
    global OUT, WORK
    arm = arm or list(STUDIES[study])[0]
    saved = (OUT, WORK, dict(BUDGET))
    OUT, WORK = ROOT / 'dry' / 'results', ROOT / 'dry' / 'work'
    if (ROOT / 'dry').exists():
        shutil.rmtree(ROOT / 'dry')
    BUDGET.update(epochs=2, flow_updates=100, fit_steps=20, rows_per_method=2)
    t0 = time.time()
    try:
        log(f'dry run of {study}/{arm} (tiny budgets; numbers are meaningless)')
        (run_heldout_arm if study == 'heldout' else run_loss_ablation_arm)(arm)
        package('dry_run_check')
    finally:
        OUT, WORK = saved[0], saved[1]
        BUDGET.clear(); BUDGET.update(saved[2])
        _dec.clear()
        shutil.rmtree(ROOT / 'dry', ignore_errors=True)
        (ROOT / 'PhyloSDF_dry_run_check.zip').unlink(missing_ok=True)
        for p in ROOT.glob('PhyloSDF_shuffle_*_latent_only.zip'):
            if p.stat().st_mtime >= t0:
                p.unlink(missing_ok=True)
    log(f'dry run passed in {time.time() - t0:.0f} s')


# ---------------------------------------------------------------- packaging and main ---------
def package(tag):
    path = ROOT / f'PhyloSDF_{tag}.zip'
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and '/surface_samples/' not in str(p) and '/rmc_work/' not in str(p) \
                    and not p.name.endswith('.tmp'):
                z.write(p, Path('results') / p.relative_to(OUT))
        for p in sorted(WORK.rglob('*')):
            text = str(p)
            if p.is_file() and '/tasks/' not in text and 'rmc_work' not in text and p.name not in ('resume.pt',) \
                    and (p.suffix in ('.json', '.npz', '.pt')):
                z.write(p, Path('work') / p.relative_to(WORK))
    return path


def download(path):
    try:
        from google.colab import files
        files.download(str(path))
    except Exception as error:
        log(f'automatic download failed ({error}); use the Files panel to download {path}')


def run(study, arms):
    if study not in STUDIES:
        raise ValueError(study)
    if study == 'heldout':
        real_skull_distances()
    for arm in arms:
        if arm not in STUDIES[study]:
            raise ValueError(f'unknown arm {arm} for {study}')
        if (OUT / arm / 'ARM_COMPLETE.json').exists():
            log(f'{arm} already complete')
        else:
            log(f'=== {study}: {arm} ===')
            (run_heldout_arm if study == 'heldout' else run_loss_ablation_arm)(arm)
        path = package(f'{study}_{arm}')
        log(f'{arm} done; saved {path} ({path.stat().st_size / 2 ** 20:.1f} MB)')
        download(path)

# ---------------------------------------------------------------- summary --------------------
# Table 5 arms from the loss-ablation run (PhyloSDF_loss_ablation_ALL.zip, A100, seed 42), for comparison only.
LOSS_ABLATION_REFERENCE = {
    'beta0':      {'label': 'no loss (beta = 0)', 'LPC_used': None, 'LPC_proxy': 0.2288, 'corr_tree': 0.1660,
                   'p_tree': 0.1178, 'recon': 0.2530, 'nn': 1.2717, 'mmd': 1.1758, 'cov': 0.555},
    'beta_proxy': {'label': 'proxy matrix (paper)', 'LPC_used': 0.9927, 'LPC_proxy': 0.9927, 'corr_tree': 0.2943,
                   'p_tree': 0.0001, 'recon': 0.2529, 'nn': 1.1961, 'mmd': 1.2171, 'cov': 0.515},
    'beta_tree':  {'label': 'dated tree', 'LPC_used': 0.9914, 'LPC_proxy': 0.3603, 'corr_tree': 0.9915,
                   'p_tree': 0.0001, 'recon': 0.2528, 'nn': 1.2233, 'mmd': 1.2875, 'cov': 0.46},
}


def _arm_row(arm):
    res = OUT / arm
    if not (res / 'LATENT_SUMMARY.json').exists():
        return None
    s = json.loads((res / 'LATENT_SUMMARY.json').read_text())
    row = {'label': f"shuffled matrix {arm[len('beta_shuf'):]}" if arm.startswith('beta_shuf') else arm,
           'LPC_used': s.get('specimen_LPC_used', s.get('specimen_LPC_proxy')), 'LPC_proxy': s.get('specimen_LPC_proxy'),
           'corr_tree': s.get('centroid_corr_tree'), 'p_tree': s.get('centroid_mantel_p_tree'),
           'recon': None, 'nn': None, 'mmd': None, 'cov': None}
    rec = res / 'metrics_reconstruction' / 'reconstruction_per_specimen.json'
    if rec.exists():
        vals = [r['mean_matched_specimen_CD'] for r in json.loads(rec.read_text())
                if r.get('mean_matched_specimen_CD') is not None]
        row['recon'] = float(np.mean(vals)) * 1e3 if vals else None
    gen = res / f'metrics_generation_seed{GEN_SEEDS[0]}' / 'summary.json'
    if gen.exists():
        for g in json.loads(gen.read_text()).get('groups', []):
            if g['cohort'] == 'unconditional_species_mixture':
                ss = g['geometry_surface_seed_summary']
                row.update(nn=ss['generated_to_reference_nn_CD_mean']['mean'] * 1e3,
                           mmd=ss['MMD_CD_reference_to_generated']['mean'] * 1e3,
                           cov=ss['COV_reference_fraction']['mean'])
    return row


def summary_table(arms=None):
    """Shuffled arms next to the Table 5 arms: LPC with the training matrix and with the proxy (specimen pairs),
    centroid correlation with the dated tree (Mantel p), reconstruction CD and species-mixture generation (x1e-3)."""
    arms = arms or list(STUDIES['shuffle_control'])
    fmt = lambda v, d=3: '   -  ' if v is None else f'{v:.{d}f}'
    lines = [f"{'matrix used in training':26s} {'LPC(used)':>9s} {'LPC(proxy)':>10s} {'r(tree)':>8s} {'Mantel p':>8s} "
             f"{'recon':>6s} {'NN-CD':>6s} {'MMD':>6s} {'COV':>5s}"]
    rows = [LOSS_ABLATION_REFERENCE[k] for k in ('beta0', 'beta_proxy', 'beta_tree')]
    rows += [r for r in (_arm_row(a) for a in arms) if r is not None]
    for r in rows:
        lines.append(f"{r['label']:26s} {fmt(r['LPC_used']):>9s} {fmt(r['LPC_proxy']):>10s} {fmt(r['corr_tree']):>8s} "
                     f"{fmt(r['p_tree'], 4):>8s} {fmt(r['recon']):>6s} {fmt(r['nn'], 2):>6s} {fmt(r['mmd'], 2):>6s} "
                     f"{fmt(r['cov'], 2):>5s}")
    text = '\n'.join(lines)
    print(text)
    (OUT / 'SHUFFLE_SUMMARY.txt').write_text(text + '\n')
    return rows
