# Species and dated-tree inputs

`specimens.csv` fixes the 100 source filenames, 24 species labels and zero-based species indices used by the archived evaluation. Keep the original capitalization and spelling.

`dated_tree_distances_Myr.csv` is the 24-by-24 matrix embedded in the original `rep_helper.py` as `TREE_MYR`, with its original `TREE_SPECIES` order. Entries are patristic distances in millions of years. The engines normalize this matrix using their archived implementation; this CSV preserves the original values.

Tree source: McTavishLab/AvesData, commit `15ddd13c81a0b5cffb37c68ba32b03ada129abfc`, [`Tree_versions/Aves_1.5/Clements2024/summary_dated_clements.nex`](https://github.com/McTavishLab/AvesData/blob/15ddd13c81a0b5cffb37c68ba32b03ada129abfc/Tree_versions/Aves_1.5/Clements2024/summary_dated_clements.nex).

Earlier component studies use the archived taxonomic proxy in the original engine. That proxy is retained as implemented for those reported experiments; it is a separate input from this dated-tree matrix.
