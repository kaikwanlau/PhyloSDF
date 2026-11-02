# Paper studies

The six engines preserve the scientific functions, constants and embedded arrays recovered from the original study notebooks. Private storage locators are replaced with local-only aliases. `SOURCE_HASHES.json` preserves archival identities and records the hashes of changed distributed files.

| Paper topic | Local study | Original engine |
| :--- | :--- | :--- |
| Proxy component comparisons and the generated-skull figure | `final` | `fm_engine.py` |
| Corrected specimen-size conditioning | `corrected-size` | `fmfix_engine.py` |
| No phylogenetic loss, proxy loss and dated-tree loss | `loss` | `rep_engine.py` |
| Strict 79-training / 21-withheld split | `heldout` | `rep_engine.py` |
| Shuffled proxy labels on all 100 skulls | `shuffled-proxy` | `shuf_engine.py` |
| Shuffled tree labels in the strict withheld split | `shuffled-tree` | `st_engine.py` and `rep_engine.py` |
| Orbit, braincase and beak measurements | `anatomy` | `anat_engine.py` |

The dated-tree main sampler comparison has its own [entry point](../tree_comparison/). Its batch size is 100; the proxy component studies use 256. Keep these experiments separate.

## Choose a study

Run from the repository root:

```bash
python code/studies/run.py list
python code/studies/run.py --help
```

Listing and help use only Python's standard library. Full execution needs the numerical packages in the repository requirements and a CUDA GPU, as in the original notebooks.

## Stage the original inputs

Full experiment archives and complete result bundles are outside this code-and-mesh release. The study launcher reads required archive filenames and SHA-256 values directly from each engine's `INPUTS` definition. Running an archived study still requires those original input ZIPs to be supplied locally in `runs/input_archives/`.

Place the 100 original processed SDF files under `data/runtime_references/processed/`, retaining their filenames. The [frozen reference manifest](../vendor/new_mesh_metrics/vendor/reference_manifest.json) records every file's local reference path and checksum. The source STL meshes included in `data/source_meshes/` are found automatically. The processed arrays preserve the original normalization and SDF sampling; they are additional runtime inputs.

The anatomy study instead requires a local checkout of [skull-morphology](https://github.com/kaikwanlau/skull-morphology) at commit `1a60c310a88574941da35ad99aacf29266aba060`, because it uses that version's measurement code as well as its meshes. Pass its location with `--skull-repo`.

Verify staged inputs before starting a long run:

```bash
python code/studies/run.py check --study loss
```

## Run

For example, run one seed of the neutral proxy residual flow:

```bash
python code/studies/run.py run --study final --arms pooled_residual --seeds 42
```

Run a complete study by omitting `--arms` and `--seeds`:

```bash
python code/studies/run.py run --study corrected-size
python code/studies/run.py run --study loss
python code/studies/run.py run --study heldout
```

The `final` defaults include neutral flow, no variance term, augmentation and calibrated temperature. The corrected-size study supplies the three current species-conditioned comparisons. The original superseded size-conditioned arms remain visible in the authentic source; the launcher does not select them by default. The shuffled-proxy release selects and accepts only the three reported controls: `beta_shuf1`, `beta_shuf2` and `beta_shuf3`.

`--inputs`, `--references` and `--workspace` accept alternative local directories. Outputs go to `runs/<study>/`; the launcher uses local inputs and disables automatic Drive fetching and Colab file downloads. It changes paths and output handling only. The original engine setup retains its checkpoint, input and environment checks.

Full studies require substantial GPU time. `setup` stages and checks the original environment, while `validate` invokes the original numerical validation. Neither is a reproduction of a full experiment.
