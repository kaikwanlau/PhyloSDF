# Mesh downloads

| Folder | Contents | STL files |
| :--- | :--- | ---: |
| [GMM](GMM/README.md) | GMM and per-species Gaussian baseline outputs | 1,720 |
| [generated](generated/README.md) | PhyloSDF pooled residual-flow outputs only | 1,260 |

The complete archive remains on GitHub: **7,305 STL files, 65.90 GB, across 12 releases**. The two folders above feature GMM and PhyloSDF outputs. Other baselines, controls, interpolations and reference meshes remain available through the [full archive CSV](catalogue.csv) and [GitHub releases](https://github.com/kaikwanlau/PhyloSDF/releases). No mesh is deleted.

## Filenames

- PhyloSDF: `generated/G_fortis_seed42_001.stl`.
- GMM: `GMM/GMM_G_fortis_seed42_001.stl` when species and seed are recorded.
- Historical GMM files omit an unrecorded seed; mixtures without a species label use `GMM_mixture_001.stl`.

The number uniquely identifies a file within its group, species and seed. It is not the original generation slot. The name and provenance index preserves the original filename, source path, experiment, sampling mode, original slot, release asset ID and SHA-256 checksum.

A species in a species-mixture filename identifies the centroid selected by the recorded generation schedule. The original conditional/mixture distinction is retained in the index.

## Download into the two folders

From the repository root:

```bash
python tools/mesh_release_names.py download --folder generated --species G.Fortis --seed 42 --limit 3
python tools/mesh_release_names.py download --folder GMM --limit 3
```

Files are saved under `downloaded_meshes/generated/` or `downloaded_meshes/GMM/` using the short names. Each download is checked against its size and SHA-256. Omit `--limit` to download all matching files. Private-repository downloads use `GH_TOKEN` or `GITHUB_TOKEN` from your local environment.

You can also use individual STL links on the catalogue pages. The STL data stays in GitHub Releases; the repository contains the catalogue and compact examples.

The short names are ready in this catalogue. Until the release-asset migration is applied, direct STL links may save the original long filename; the command above saves the short filename.

## Coordinates and source data

STL is unitless. Renaming does not repair, rescale or otherwise change geometry. Consult the source manifests and experiment fields for coordinates and extraction settings. The [100 original prepared inputs](../data/source_meshes/README.md) remain in `data/source_meshes/` with their original attribution and data terms. See the [compact output notes](../outputs/README.md) for figure and anatomical examples.
