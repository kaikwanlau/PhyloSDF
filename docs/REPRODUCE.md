# Reproduce the paper

## Environment

Install a CUDA-enabled PyTorch build suitable for your GPU, then install:

```bash
python -m pip install -r requirements-research.txt
```

The archived main comparison recorded Python 3.13.16, PyTorch 2.11.0+cu130, CUDA 13.0, NumPy 2.1.3, trimesh 4.7.4 and scikit-image 0.25.2 on an A100 40 GB. Other builds may differ numerically. `requirements-research.txt` is a practical dependency list; the archived runtime above records the actual study environment.

## Choose the experiment

| Paper topic | Entry point | Key distinction |
| :--- | :--- | :--- |
| Main sampler comparison and Fig. 6a | [`code/tree_comparison/run.py`](../code/tree_comparison/) | One dated-tree representation, flow batch 100, seeds 42–44, four matched samplers |
| Generated-skull examples and Fig. 6b | [`code/studies/run.py`](../code/studies/) `final` | Proxy representation, batch 256, neutral residual flow and component comparisons |
| Corrected size inputs | `corrected-size` | Three current species-conditioned comparisons |
| Phylogenetic supervision | `loss` and `shuffled-proxy` | Matched representation training on all 100 skulls |
| Strict withheld species | `heldout` and `shuffled-tree` | 79 training / 21 withheld skulls; no withheld centroid except the labelled oracle |
| Orbit, braincase and beak measurements | `anatomy` | Padded measurement-grid extraction in millimetres |

The original sources include historical helper functions used by the retained engines. The local launchers select the studies above. Two superseded wrong-size rows are retained in the authentic source but are excluded from the default final-study launcher.

## Original and processed inputs

All 100 source STL meshes are included in [`data/source_meshes/`](../data/source_meshes/). Every file matches the raw reference hash in the frozen evaluation manifest. They are prepared surface meshes from the source study, before PhyloSDF-specific watertight/SDF processing.

Exact replay also uses the original processed NPZ records, which preserve saved centroids, scales and SDF samples. Their local paths and checksums are listed in [`reference_manifest.json`](../code/vendor/new_mesh_metrics/vendor/reference_manifest.json). Place them in `data/runtime_references/processed/` with the original filenames. Do not silently replace them with newly sampled SDF points.

This release provides code, mesh files and the paper link; full experiment archives and complete result bundles are outside its scope. The study launcher reads required archive filenames and SHA-256 checksums from the `INPUTS` definition in each engine under [`code/studies/`](../code/studies/). Running the archived studies still requires those original input ZIPs in `runs/input_archives/`, together with the processed records and checkpoints needed by the selected study.

### Dated-tree main comparison

These archives are not distributed with this release. To run the archived comparison, supply authorized local copies using the required names:

| Local name | Required contents |
| :--- | :--- |
| `loss_ablation.zip` | Paired representation-loss archive |
| `matched_inputs.zip` | Matched CFM source and inputs |
| `mesh_inputs.zip` | Frozen mesh extraction and evaluation source |

Stage a reference directory with `raw_stl/` containing copies of the included STL files and `processed/` containing the verified original NPZ files. Then:

```bash
python code/tree_comparison/run.py preflight --base work/tree_comparison \
  --archives /path/to/tree_archives --references /path/to/reference_root
python code/tree_comparison/run.py run --base work/tree_comparison
python code/tree_comparison/run.py postprocess --base work/tree_comparison
```

The source checks the dated-tree decoder hash, original latent ordering, reference transforms and protocol before running. Use a fresh base directory for a new experiment. See the [tree instructions](../code/tree_comparison/README.md) for the exact archive and model checksums.

### Component and control studies

```bash
python code/studies/run.py list
python code/studies/run.py check --study loss
python code/studies/run.py run --study final --arms pooled_residual --seeds 42
```

The [study guide](../code/studies/README.md) gives the other study names, local paths and anatomy checkout requirement. Full runs retain the original budgets. Input checking and the small quick-start preview do not train a model.

## Scope of the recovered release

Scientific calculations and model settings are retained. Private storage locators are removed from distributed files; source manifests preserve archival hashes and record distributed-file hashes where metadata redaction changes bytes. The local launchers stage user-supplied inputs and handle outputs locally. The main sampler comparison is conditional on one trained representation; its three generator seeds do not repeat representation training.

Selected original code for the earlier LOSO and 79/21 analyses still cited in the paper is retained in [`code/historical/`](../code/historical/). That folder documents the surviving sources, shared dependencies and missing runtime components. The LOSO script computes a latent-space diagnostic without retraining or decoding meshes, despite its inaccurate original opening description. The older 79/21 source subset is not a complete replay workflow.

Historical lowest-loss decoder and initial conditioned-generator workflows also have unrecovered configurations or outputs. The paper records that the DDPM baseline's inputs/configuration were not recovered; this release does not claim a reproducible DDPM result.
