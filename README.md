# Pharmacophore-Modeling-3D-Similarity-Tutorial

**Complete ligand-based pharmacophore modeling in Python with 3D similarity plots (PNG 600 dpi + SVG) and per-compound reports.**

Set a reference compound, drop your test compounds into `smiles.smi`, run one script: you get pharmacophore features per
compound, distance matrices, feature distributions, detailed similarity reports and high-quality **3D similarity plots**.

> بالعربي: كود كامل لعمل Pharmacophore modeling ورسم تشابه ثلاثي الأبعاد لمجموعة مركبات مقابل مركب مرجعي.

## Use

1. Edit the configuration section at the top of `pharmacophore_3d_similarity.py` (reference SMILES + name).
2. Put test compounds in `smiles.smi`.
3. `python pharmacophore_3d_similarity.py`

Outputs: 3D similarity plots (PNG 600 DPI + SVG), individual pharmacophore analysis, distance matrices, feature distributions, text reports.
Requires RDKit, numpy, matplotlib.


## Repository map

- `pharmacophore_3d_similarity.py`

## Author

Yaseen Saleem Hamdoon - Pharmaceutical chemist & computational drug designer, Lecturer in Organic Pharmaceutical Chemistry, Al-Kitab University (Kirkuk, Iraq).

## License

MIT - see `LICENSE`. Third-party data/tools keep their own licenses.
