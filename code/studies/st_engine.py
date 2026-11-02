# st_engine.py -- shuffled-tree control for the held-out-species study (PhyloSDF revision).
#
# Imports rep_engine.py (unchanged bytes; the engine that produced the held-out study, ho_beta0/proxy/tree) and adds
# arms ho_beta_shuffle1..3: identical to ho_beta_tree (79 training skulls, beta = 5e-3, 2,000 epochs, same seed and
# data order) except that the dated tree's tip labels are randomly permuted (same distance distribution, wrong
# species; permutation seeds fixed in advance). Question: does the dated tree's benefit for UNSEEN species (better
# reconstruction of withheld skulls, better centroid prediction) come from the phylogeny itself, or would any
# tree-shaped regulariser do the same?
#
# Two changes that leave every value unchanged: the Chamfer nearest-neighbour search uses the frozen evaluator's exact
# KD-tree (multithreaded, as validated in the final-model ablation v2) instead of the slow GPU kernel, and an uploaded
# zip from an earlier session lets a new session continue.
# Google Drive is used READ-ONLY (inherited from rep_engine).
import json, time, zipfile
from pathlib import Path
import numpy as np
import rep_engine as rep

SHUFFLES = (1, 2, 3)
PERM_SEED_BASE = 20261020
UPLOAD_DIR = Path('/content')
log = rep.log


def permutation(k):
    return np.random.default_rng(PERM_SEED_BASE + k).permutation(24)


_original_matrix = rep.matrix


def matrix(kind):
    if str(kind).startswith('tree_shuffle'):
        p = permutation(int(str(kind)[len('tree_shuffle'):]))
        T = _original_matrix('tree')
        return T[np.ix_(p, p)]           # species i sits at tip p[i] of the dated tree
    return _original_matrix(kind)


rep.matrix = matrix
for k in SHUFFLES:
    rep.STUDIES['heldout'][f'ho_beta_shuffle{k}'] = {'beta': 5e-3, 'matrix': f'tree_shuffle{k}', 'specimens': 'train79'}

# 'own_knn': the unseen species' centroid predicted from the 3 nearest training species of the arm's OWN matrix
CURRENT = {'matrix': 'tree'}
_original_predict = rep.predict_centroids


def predict_centroids(train_codes, train_species, heldout_species, names_sp):
    out = _original_predict(train_codes, train_species, heldout_species, names_sp)
    M = matrix(CURRENT['matrix'])
    present = sorted(set(train_species.tolist()))
    C = {s: train_codes[train_species == s].mean(0) for s in present}
    for h in heldout_species:
        near = sorted(present, key=lambda s: (M[h, s], names_sp[s]))[:3]
        w = np.array([1.0 / (M[h, s] + 1e-6) for s in near])
        out[h]['own_knn'] = {'centroid': (sum(wi * C[s] for wi, s in zip(w, near)) / w.sum()).astype(np.float32),
                             'neighbours': [names_sp[s] for s in near], 'distances': [float(M[h, s]) for s in near],
                             'weights': (w / w.sum()).tolist(), 'matrix': CURRENT['matrix']}
    return out


rep.predict_centroids = predict_centroids
rep.PREDICTORS = tuple(rep.PREDICTORS) + ('own_knn',)
rep.METHOD_NAMES = tuple(rep.METHOD_NAMES) + ('ho_gen_own_knn', 'ho_centroid_own_knn')

_original_run_arm = rep.run_heldout_arm


def run_heldout_arm(arm):
    CURRENT['matrix'] = rep.STUDIES['heldout'][arm]['matrix']
    _original_run_arm(arm)
    own_summary(arm)


rep.run_heldout_arm = run_heldout_arm


def own_summary(arm):
    """LPC of the trained codes with the arm's OWN matrix and with the true dated tree (Mantel permutations)."""
    from scipy.spatial.distance import pdist, squareform
    from scipy.stats import pearsonr
    out = rep.OUT / arm / 'OWN_MATRIX_SUMMARY.json'
    if out.exists():
        return
    codes, species, keep, names_train = rep.load_codes(arm)
    own, tree = matrix(rep.STUDIES['heldout'][arm]['matrix']), _original_matrix('tree')
    present = np.array(sorted(set(species.tolist())))
    C = np.stack([codes[species == s].mean(0) for s in present])
    L = squareform(pdist(np.asarray(C, np.float64)))
    iu = np.triu_indices(len(present), 1)
    rng = np.random.default_rng(0)
    summary = {'arm': arm, 'matrix': rep.STUDIES['heldout'][arm]['matrix'],
               'permutation': permutation(int(arm[-1])).tolist() if 'shuffle' in arm else None,
               'specimen_LPC_own_matrix': rep.pair_correlation(codes, species, own)}
    for label, M in (('own', own), ('true_tree', tree)):
        sub = M[np.ix_(present, present)]
        r = pearsonr(L[iu], sub[iu])[0]
        null = []
        for _ in range(9999):
            p = rng.permutation(len(present))
            null.append(pearsonr(L[iu], sub[np.ix_(p, p)][iu])[0])
        summary[f'centroid_corr_{label}'] = float(r)
        summary[f'centroid_mantel_p_{label}'] = float((1 + np.sum(np.asarray(null) >= r)) / 10000)
    tri = np.triu_indices(len(present), 1)
    summary['own_matrix_vs_true_tree_corr_training_species'] = float(
        pearsonr(own[np.ix_(present, present)][tri], tree[np.ix_(present, present)][tri])[0])
    rep.write_json(out, summary)
    log(f'{arm}: LPC with its own (shuffled) matrix {summary["specimen_LPC_own_matrix"]:.3f}; centroid r with the true '
        f'tree {summary["centroid_corr_true_tree"]:.3f} (p={summary["centroid_mantel_p_true_tree"]:.4f})')


# ---- exact KD-tree Chamfer distance (same values as the frozen evaluator; only the query is multithreaded)
def kd_cd(left, right, left_tree=None, right_tree=None):
    from scipy.spatial import cKDTree
    left_tree = cKDTree(left) if left_tree is None else left_tree
    right_tree = cKDTree(right) if right_tree is None else right_tree
    value = float(np.square(left_tree.query(right, workers=-1)[0]).mean()
                  + np.square(right_tree.query(left, workers=-1)[0]).mean())
    if not np.isfinite(value) or value < 0:
        raise ValueError('Invalid Chamfer value')
    return value


rep.gpu_cd = kd_cd       # rep.vendor() and rep.real_real_cd look this name up when they run


def restore():
    """Continue from the newest PhyloSDF_heldout_*.zip uploaded to /content (results and trained decoders)."""
    zips = sorted(UPLOAD_DIR.glob('PhyloSDF_heldout_*.zip'), key=lambda p: p.stat().st_size)
    if not zips:
        return 0
    path, n = zips[-1], 0
    with zipfile.ZipFile(path) as z:
        for member in z.namelist():
            parts = Path(member).parts
            if member.endswith('/') or '..' in parts or parts[0] not in ('results', 'work'):
                continue
            target = (rep.OUT if parts[0] == 'results' else rep.WORK) / Path(*parts[1:])
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(z.read(member))
                n += 1
    done = [a for a in rep.STUDIES['heldout'] if (rep.OUT / a / 'ARM_COMPLETE.json').exists()]
    log(f'restored {n} files from {path.name}; complete arms: {", ".join(done) or "none"}')
    return n


def summary():
    rows = {}
    for arm in [f'ho_beta_shuffle{k}' for k in SHUFFLES]:
        rec = rep.OUT / arm / 'metrics' / 'reconstruction_per_specimen.json'
        if rec.exists():
            vals = [r['mean_matched_specimen_CD'] for r in json.loads(rec.read_text())
                    if r.get('role') == 'heldout_fit' and r.get('mean_matched_specimen_CD') is not None]
            rows[arm] = float(np.mean(vals)) * 1e3
    log('withheld-skull reconstruction CD x1e-3 (earlier run: no loss 0.525, dated tree 0.487): '
        + json.dumps({k: round(v, 3) for k, v in rows.items()}))
    return rows