# DrugBAN Paper Replication

Reproduce DrugBAN paper (Bai et al., 2023) benchmarks with our AnchorDrugBAN model.

## What this does

Trains and evaluates on the **exact same datasets and splits** as the DrugBAN paper:
- **BindingDB** (49K interactions) — random + cluster split
- **BioSNAP** (27K interactions) — random + cluster split
- **Human** (6.7K interactions) — random + cold pair split

## Models compared

| Model | Description |
|-------|-------------|
| `drugban` | Vanilla DrugBAN baseline (GIN + CNN + bilinear attention) |
| `anchor_drugban` | AnchorDrugBAN with drug-centric Tanimoto anchor retrieval |
| `drugban_anchor_subset` | DrugBAN trained/tested on same samples as anchor model |
| `anchor_drugban_oracle` | AnchorDrugBAN with perfect (oracle) anchors — upper bound |
| `drugban_oracle_subset` | DrugBAN trained/tested on same samples as oracle model |

## Quick start

```bash
# 1. Setup
bash reproduce/drugban_paper/01_setup.sh

# 2. Download data
bash reproduce/drugban_paper/02_fetch_data.sh

# 3. Run in-domain experiments (random splits)
bash reproduce/drugban_paper/03_run_indomain.sh

# 4. Run cross-domain experiments (cluster + cold splits)
bash reproduce/drugban_paper/04_run_crossdomain.sh

# 5. Summarize all results
bash reproduce/drugban_paper/05_summarize.sh
```

## Configuration

Override defaults via environment variables:

```bash
EPOCHS=100 PATIENCE=20 SEEDS="0,1,2,3,4" bash reproduce/drugban_paper/03_run_indomain.sh
```

## Resumability

All runs are crash-safe. Results are saved to CSV after each run.
Re-running the same script skips already completed (dataset, split, model, seed) combinations.

## Anchor strategy

**Drug-centric retrieval**: for query `(drug_q, protein_q)`:
1. Find `drug_a` = most Tanimoto-similar drug in training positives (Morgan FP, radius=2)
2. Get `protein_a` = a protein that `drug_a` binds (Y=1), different from `protein_q`
3. Model compares binding patterns: `drug_q` vs `protein_a` (anchor) vs `protein_q` (query)

No self-anchors: `drug_a != drug_q` AND `protein_a != protein_q`.

## Cross-domain protocol

For cluster splits, we follow the paper's protocol:
- Train on **source domain only** (labeled)
- Target domain train data is NOT used (paper uses it for CDAN adversarial alignment; we don't implement CDAN)
- Test on target domain held-out set

This is a fair comparison with the paper's vanilla DrugBAN baseline.
