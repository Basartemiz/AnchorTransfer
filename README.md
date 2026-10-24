# Anchor Transfer Learning for Drug-Target Affinity

Code for the paper *Anchor Transfer Learning for Cross-Dataset Drug-Target
Affinity Prediction* ([Zenodo, v5](https://doi.org/10.5281/zenodo.21902290)).

## Idea

To predict how strongly a drug binds a protein, the model also gets an
**anchor**: a known protein that binds a similar drug.

## Experiments

| # | Anchor model | Baseline |
|---|---|---|
| 1 | AnchorMLP | ESM-DTA |
| 2 | AnchorDrugBAN | DrugBAN |
| 3 | ConciseAnchor | CoNCISE |
| 4 | ConciseAnchor (activity cliffs) | CoNCISE |

Trained on DTC and BindingDB, tested on Davis.

## Folders

- `experiments/` — the paper's experiments
- `src/anchor_transfer/` — the models
- `revised_paper/` — paper source

## Reproduce

See [`REPRODUCE.md`](REPRODUCE.md).

> Note: the v5 PDF describes the anchor retrieval slightly wrong. The corrected
> text is in `revised_paper/`.

## Citation

```bibtex
@misc{temiz2026anchor,
  title  = {Anchor Transfer Learning for Cross-Dataset Drug-Target Affinity Prediction},
  author = {Temiz, Ba{\c{s}}ar},
  year   = {2026},
  doi    = {10.5281/zenodo.21902290}
}
```
