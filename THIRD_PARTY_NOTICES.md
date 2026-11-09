# License scope and third-party materials

The repository's MIT License applies to original source code and associated
software documentation that the Ligand State Recovery contributors have the
right to license. It does not relicense incorporated third-party material or
grant rights in data, weights, model outputs, manuscript content, or external
executables.

- **BondNet companion project.** Reused BondNet source retains its MIT notice.
  See [the upstream release](https://github.com/zxc8848/BondNet/tree/v4.1-revision)
  and the accompanying `LICENSES/BondNet-v4.1-MIT.txt`. The separately archived
  companion weights are subject to the terms of the
  [BondNet v4.1 data and model record](https://doi.org/10.5281/zenodo.23080342).
- **Dependencies.** PyTorch, RDKit, SciPy, NumPy, Open Babel and Matplotlib are
  installed separately (see `environment.yml`) and retain their own licenses.
  This repository does not redistribute those libraries.
- **Knodle.** The comparison wrapper expects a separately acquired Knodle
  executable. The executable is not included and is not licensed under this
  repository's MIT license.
- **Benchmark and training materials.** PDBBind poses, training records,
  training caches, the new completion-head weights and the training-derived
  catalogue are not distributed by this code release. Refer to the separate
  data records and upstream providers for terms applicable to their materials.
