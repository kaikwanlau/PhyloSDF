# Released mesh outputs

| Collection | Model and extraction | Coordinate frame |
| :--- | :--- | :--- |
| [`figure_meshes/`](figure_meshes/) | Proxy component-study final residual flow, T=1, seed 42; 256³ grid over [-1,1]³; slots 000–002 of four focal species | Normalized decoder coordinates |
| [`anatomical_meshes/`](anatomical_meshes/) | Two saved proxy anatomical-audit examples, seed 42, slot 000; padded measurement grid | Millimetres |

Each collection's `manifest.json` records the original archive, member path, species, seed, slot, coordinate frame and SHA-256. NPZ files preserve the original saved bytes and contain `vertices` and `faces`. STL examples export the same float32 triangle coordinates without smoothing, repairing, filtering or changing scale.

Load a mesh with NumPy:

```python
import numpy as np
from pathlib import Path

path = next(Path("outputs/figure_meshes").rglob("*.npz"))
with np.load(path, allow_pickle=False) as mesh:
    vertices, faces = mesh["vertices"], mesh["faces"]
```

To export any released NPZ as an STL:

```bash
python tools/export_stl.py path/to/mesh.npz output/mesh.stl
```

The saved high-resolution figure cohort is distinct from the dated-tree main sampler comparison. The two anatomical meshes are archived measurement examples; they do not claim to be the exact re-extractions rendered in Figures 8–9. For example, the archived C. parvulus mesh has 3,832 vertices while the manuscript describes a displayed 3,452-vertex re-extraction.
