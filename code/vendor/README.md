# Frozen implementation dependencies

These are the exact implementation files loaded by the paper's study engines:

- `matched_cfm/`: fixed-update flow training, deterministic resume state, DeepSDF decoder, flow network and latent augmentation.
- `new_mesh_metrics/`: frozen mesh evaluator and the decoder, extraction and checkpoint-loading functions it shares with the study code.

`SOURCE_HASHES.json` records each included file and its archival source identity. Private storage locators in metadata are redacted; distributed-file hashes are recorded where that changes file bytes. The source manifests retain the broader archive inventory; not every archive asset is bundled here. Checkpoints, prepared training arrays and processed reference inputs are staged separately as described in [Paper studies](../studies/README.md).

These modules support the study entry points. The original standalone archival utilities and checkpoint-audit CLI are outside this dependency subset.

The shared file `matched_cfm/vendor/diffusion.py` supplies `PhyloFlowNet` and `PhyloGMM`, both used in the paper. Its unused DDPM compatibility classes are retained to preserve the exact source bytes. They are not selected by the released study launchers or presented as a separate experiment.
