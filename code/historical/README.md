# Historical analyses cited in the paper

These selected original sources support the earlier LOSO and 79/21 analyses still discussed in the main text. They remain separate from the current [representation studies](../studies/) and [dated-tree sampler comparison](../tree_comparison/). `SOURCE_HASHES.json` retains source filenames and scientific-file checksums. Personal storage URLs and identifiers are omitted from the distribution.

## Earlier LOSO diagnostic

`loso/12_leave_one_species_out.py`, its original `models/utils.py` and `config.yaml` are preserved. The package marker `models/__init__.py` is new; it avoids importing unrelated legacy models.

The original script's opening description is inaccurate: its body **does not retrain a model or compute mesh Chamfer distances**. It loads codes trained on all species, predicts a centroid from three proxy neighbours, adds Gaussian residual noise, and compares **latent L2 distances**. It does not invoke the learned flow or decode meshes.

Replay requires the original `checkpoints/deepsdf/` latent codes, species indices and proxy matrix; `checkpoints/diffusion/` centroids and residual statistics; and the original processed metadata. These artifacts are not supplied here. The source configuration is preserved without claiming it proves the historical runtime configuration.

## Earlier 79/21 reconstruction study

`79_21/decoder_training.py`, `generator.py` and `metrics.py` were recovered individually from the archived code companion and verified against their embedded original hashes. Identical shared `deepsdf.py` and `vendor_metrics.py` implementations already reside in `code/vendor/`; their paths are recorded in the hash manifest.

This is a different retrospective study from the current strict-holdout runs. `common.py`, the original latent-fitting and decoding drivers, and the sealed-SDF evaluation source were not recovered. Processed samples, immutable protocols, checkpoint locks and trained/fitted artifacts also remain external requirements. The saved-source subset therefore **does not provide an independently complete replay** of the earlier reconstruction result. No missing runtime configuration has been reconstructed or substituted.
