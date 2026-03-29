# Domain-Native DeepDTA: Anchor Prediction Framework

**Date:** 2026-03-29
**Status:** Approved
**Branch:** alphafold

## Problem

Current DeepDTA was trained on full protein sequences (~500-1000 residues). When we feed shorter domain sequences (~50-300 residues) from structural anchors, MSE degrades due to input distribution mismatch. We need a DeepDTA trained from scratch on domain sequences so anchor-based predictions are native.

## Training Data

- **Source:** BindingDB full dump — proteins that have domains in our 48K model_organism_domains database
- **Binding domain identification (hybrid priority):**
  1. UniProt binding site annotations → find which domain overlaps with annotated residues
  2. P2Rank largest pocket volume fallback (run on each domain PDB)
  3. Largest domain as last resort
- **Training pairs:** (binding_domain_sequence, SMILES) → pKi
- **Hold-out:** 203 benchmark proteins excluded from training

## Architecture

DeepDTA with shorter-input adaptations:

```
Domain seq (max 500)  →  Embedding(26, 128)  →  Conv1d[3,4,6] × 32 filters  →  MaxPool  →  concat
SMILES    (max 100)   →  Embedding(52, 128)  →  Conv1d[4,6,8] × 32 filters  →  MaxPool  →  concat
                                                                                              ↓
                                                                              FC(192→1024) → FC(1024→512) → FC(512→1)
```

Changes from standard DeepDTA:
- `PROTEIN_MAX_LEN`: 1000 → 500
- Protein CNN kernels: [4, 6, 8] → [3, 4, 6]
- Drug CNN, MLP, dropout: unchanged

## Inference Pipeline

1. For query IDP: generate conformations from IDRome
2. Foldseek search conformations against domain database
3. Pick single best anchor (highest qtmscore, threshold ≥ 0.6)
4. Feed anchor's domain sequence to domain-native DeepDTA
5. Output: predicted pKi (no aggregation, no weighting)

## Data Pipeline

1. **Download BindingDB** bulk data, filter to Ki/Kd measurements with valid UniProt ID + SMILES
2. **Convert to pKi:** pKi = 9 - log10(Ki_nM), clip to [3.0, 12.0]
3. **Identify binding domains** for each protein (UniProt → P2Rank → largest)
4. **Extract binding domain sequences** from PDB files (map domain AA to full protein positions)
5. **Build training CSV:** binding_domain_sequence, SMILES, pKi
6. **Exclude** 203 benchmark proteins from training

## Binding Domain Identification

### Step 1: UniProt Annotations
- Fetch UniProt JSON for each protein via API (`/uniprot/{uid}.json`)
- Extract features of type `BINDING` or `ACT_SITE` with residue ranges
- Map each domain's sequence to the full protein to get domain residue ranges
- Domain with most overlap to binding site annotations wins
- Confidence: high

### Step 2: P2Rank Fallback
- For proteins without UniProt binding annotations
- Run P2Rank on each domain PDB
- Domain with the largest pocket volume wins
- Confidence: medium

### Step 3: Largest Domain Fallback
- For proteins where P2Rank finds no pockets
- Pick the domain with the most residues
- Confidence: low

## Evaluation

Same 203-protein benchmark, three-way comparison:
1. **Baseline:** Seq-only full-protein DeepDTA (existing)
2. **Anchor + full-seq:** Anchor pipeline with full protein DeepDTA (current eval)
3. **Anchor + domain-native:** Anchor pipeline with domain-native DeepDTA (new)

Metrics: CI, MSE, Pearson r — stratified by disorder quartile and TM quartile.

## File Structure

```
scripts/build_domain_training_data.py    — BindingDB download + binding domain identification
scripts/train_domain_deepdta.py          — Train domain-native DeepDTA
scripts/evaluate_anchor_dta_domain.py    — Eval with domain-native model
src/idr_gat/data/binding_domain.py       — UniProt + P2Rank binding domain identification
src/idr_gat/model/domain_deepdta.py      — DeepDTA with domain-adapted architecture
```
