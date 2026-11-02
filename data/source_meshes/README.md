# Original source skull meshes

These are the 100 prepared STL skulls of Darwin's finches and their relatives
used as the source collection for PhyloSDF (24 species). They are remeshed
surface meshes, rather than raw micro-CT image volumes.

The files are unchanged copies of
[`data/DF_and_their_relatives/`](https://github.com/kaikwanlau/skull-morphology/tree/1a60c310a88574941da35ad99aacf29266aba060/data/DF_and_their_relatives)
from **skull-morphology**, commit
`1a60c310a88574941da35ad99aacf29266aba060`.
[`manifest.csv`](manifest.csv) records each filename, byte count, Git blob
SHA and SHA-256 checksum. Original filenames, capitalization and spelling
are retained.

The bird scans were acquired and processed in Tokita et al. (2017),
*Cranial shape evolution in adaptive radiations of birds: comparative
morphometrics of Darwin's finches and Hawaiian honeycreepers*;
Al-Mosleh et al. (2021), *Geometry and dynamics link form, function, and
evolution of finch beaks*; and Mosleh et al. (2023),
*Beak morphometry and morphogenesis across avian radiations*.
They were remeshed for the source skull-morphology study.

Preserve the source attributions and the original datasets' reuse terms.
The source repository's Apache 2.0 code license does not grant a new
license for all third-party scans. See its
[data provenance notes](https://github.com/kaikwanlau/skull-morphology/blob/1a60c310a88574941da35ad99aacf29266aba060/data/README.md).

These source meshes precede PhyloSDF's watertight conversion and SDF
sampling. Use the processed-input workflow documented in the main README
for model training; do not treat the two mesh stages as interchangeable.

