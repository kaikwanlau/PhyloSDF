# Dated-tree sampler comparison

These files implement the revised paper's main sampler comparison: one frozen dated-tree DeepSDF representation, four samplers, seeds 42–44, and 140 mesh attempts per sampler and seed. The flow uses batch size 100. Earlier component studies use a different representation and batch size 256.

Scientific calculations, model settings and embedded arrays are retained. Private storage locators in the sampler and helper are replaced with local-only aliases; `SOURCE_HASHES.json` keeps archival identities alongside distributed-file hashes. `run.py` stages local inputs and disables remote uploads. Historical helper entry points remain for source context.

## Inspect the protocol

From the repository root:

```bash
python code/tree_comparison/run.py protocol
python code/tree_comparison/tree_comparison_postprocess.py --synthetic-checks
```

The second command checks the 1-NNA calculation, including self exclusion, condensed-distance ordering and ties. Both commands need NumPy; full execution also needs CUDA PyTorch, SciPy, `trimesh==4.7.4` and `scikit-image==0.25.2`.

## Reproduce the comparison

Use the original archived inputs so that checkpoint, ordering and normalization checks remain valid:

| Local archive name | Required contents |
| --- | --- |
| `loss_ablation.zip` | Paired dated-tree decoder, latent codes and representation training records |
| `matched_inputs.zip` | Flow network source, resume helpers and ordered specimen metadata |
| `mesh_inputs.zip` | Frozen decoder, extraction and evaluation source; reference manifest |

The archive SHA-256 values are recorded in `tree_sampler_comparison.ARCHIVES`; individual source and model identities are in `SOURCE_HASHES.json`. These archives are experiment inputs, not additional mesh outputs. Do not substitute proxy-representation checkpoints or newer copies with different bytes.

Provide a reference directory containing `raw_stl/<specimen>.stl` and `processed/<specimen>.npz`, exactly as listed in the archived `reference_manifest.json`. The processed records supply each original saved centroid and scale. Source STL meshes alone do not replace those records.

```bash
python code/tree_comparison/run.py preflight --base work/tree_comparison \
  --archives /path/to/archives --references /path/to/reference_root
python code/tree_comparison/run.py run --base work/tree_comparison
python code/tree_comparison/run.py postprocess --base work/tree_comparison
```

Use a new `--base` directory for a new run. Preflight verifies the paired model and input identities, performs a genuine 256³ extraction, and replays a saved reconstruction metric. The run writes local meshes and metrics; postprocessing adds 1-NNA and descriptive mean/sample standard deviation across generator seeds. No remote upload or authentication occurs through `run.py`.

The generator seeds repeat sampling and flow training conditional on one representation. They do not repeat DeepSDF training or establish generalization to unseen specimens.
