# Applying the mesh filenames

The reviewed plan is [`names.csv`](names.csv): 1,720 GMM assets and 1,260 PhyloSDF pooled residual-flow assets. It records every old/new filename, immutable release asset ID, byte count, SHA-256 and source path. The 4,325 other meshes remain unchanged in the [full archive](catalogue.csv).

The catalogue and download helper already use the short names. The release assets retain their existing names until the migration is run. This lets the proposed grouping and mapping be reviewed before changing live download URLs.

## Check

After the change is merged, open **Actions → Apply reviewed mesh filenames → Run workflow**. Leave **apply** unchecked for a read-only identity and collision check. Local equivalents are:

```bash
python tools/mesh_release_names.py validate
python -m unittest discover -s tools -p test_mesh_release_names.py
python tools/mesh_release_names.py migrate
```

The last command uses a local `GH_TOKEN` or `GITHUB_TOKEN` to read private release metadata. Do not paste credentials into an issue, pull request or catalogue.

## Apply

Run the same workflow on **main** with **apply** checked. It changes release-asset names only, verifies the existing content identities, and publishes refreshed catalogue links after each completed batch. It never downloads and re-exports the STL geometry, deletes assets or changes repository visibility.

The workflow uses two sequential jobs and paces writes to respect GitHub's API limits, so the full operation takes several hours. During a batch, direct links using a renamed asset's old filename may stop working until that batch's catalogue refresh is published. The authenticated download helper uses immutable asset IDs and remains independent of filename changes.

If interrupted, rerun the same reviewed plan: assets already renamed with the expected content are accepted and skipped. A changed asset, name collision or changed plan stops the operation. Catalogue publication uses a normal fast-forward commit and preserves unrelated repository files.

Keep `original_filename`, original source paths and source manifest references in the index after migration; they are provenance, not current download filenames.
