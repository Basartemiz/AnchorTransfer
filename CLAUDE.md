# IDR-GAT Project Instructions

## Workflow Requirements

- **Always use /superpowers:brainstorming** after the user asks for something — brainstorm and align BEFORE implementing
- **Always use /controlWithCodex** after completing significant code changes to review for bugs, logic errors, and correctness
- **Always use /handleToCodex** for complex coding tasks (large refactors, multi-file features, tricky algorithms, intricate bug fixes)
- These are the primary tools for implementation and review — use them most of the time

## Code Safety

- **NEVER modify core model files** — create new files for architecture changes (backward compatibility)
- Core files that must not be modified: `src/idr_gat/model/contrastive.py`, `src/idr_gat/model/protein_encoder.py`, `src/idr_gat/model/affinity.py`, `src/idr_gat/model/drug_encoder.py`
- **NEVER delete** `metadata.json`, `similarity_cache.npz`, `3di_seqs.tsv` — run `bash docs/reference/backup_caches.sh [tag]` before config changes

## Evaluation Pipelines

- **Contrastive models** -> `scripts/evaluate_reachability.py` (AUROC, AUPRC, MRR, Hit@k)
- **Affinity models** -> `scripts/evaluate_affinity_benchmark.py` (CI, MSE, Pearson r)
- **Contrastive validation** -> `scripts/evaluate.py` (on val split)
- Always explore the repo and understand existing code before making changes

## Project Structure

- `src/idr_gat/` — core library (config, data, graph, model, training, evaluation)
- `scripts/` — focused Python scripts for training and evaluation
- `reproduce/` — numbered shell scripts for paper reproduction
- `baselines/` — comparison models
- `tests/` — test suite

## Remote Servers

- A40: `ssh root@194.68.245.215 -p 22078 -i ~/.ssh/id_ed25519` — code at `/workspace/IDP-work/`
