# Binding Site Anchor Pipeline

**Date:** 2026-03-29
**Status:** Approved
**Branch:** alphafold

## Problem

Current pipeline uses whole AlphaFold domains as Foldseek targets and graph nodes. A structurally similar domain doesn't guarantee a similar binding site — this adds noise to the anchor-based DTA prediction. The model should learn that structurally similar **binding sites** bind similar drugs.

## Approach

Replace domain nodes with binding site nodes. Use P2Rank to predict pockets on each AlphaFold domain, extract one PDB per pocket (score > 0.5), build a binding-site-only graph. Increase anchor TM threshold to 0.9 (now realistic since binding sites are similar-sized regions).

## Pipeline

```
AlphaFold Domain PDBs
       │
  P2Rank pocket prediction (score > 0.5)
       │
  One PDB per pocket (pocket residues only)
  Domain with 3 pockets → 3 binding site PDBs
       │
  Foldseek 3Di encoding + all-vs-all search
       │
  Graph: nodes = binding sites, edges = structural similarity
  Node features: 3Di + ESM-2 (binding site sequence)
       │
  Anchor lookup: IDP conformation → Foldseek → binding site DB
  TM threshold ≥ 0.9
```

## P2Rank Integration

### Installation
P2Rank is a Java-based tool. Download the binary release:
```bash
wget https://github.com/rdk/p2rank/releases/download/2.4.2/p2rank_2.4.2.tar.gz
tar xzf p2rank_2.4.2.tar.gz
```
Binary at `p2rank_2.4.2/prank`

### Running
```bash
prank predict -f domain.pdb -o output_dir/ -threads 8
```

### Output
- `output_dir/domain.pdb_predictions.csv`: ranked pockets with columns:
  - `name`: pocket name (pocket1, pocket2, ...)
  - `score`: P2Rank confidence score
  - `residue_ids`: comma-separated residue IDs (e.g., "A_42, A_43, A_44")
  - `center_x, center_y, center_z`: pocket center coordinates

### Pocket Filtering
- Keep pockets with `score > 0.5`
- Minimum 10 residues per pocket (smaller pockets unreliable for Foldseek)
- Maximum 5 pockets per domain (diminishing returns)

## Binding Site PDB Extraction

For each qualifying pocket:
1. Parse residue IDs from P2Rank output
2. Extract matching ATOM records from domain PDB (Cα only for Foldseek)
3. Write to `{domain_name}_pocket{N}.pdb`

Metadata saved as JSON:
```json
{
  "A0A5B7_d0_pocket1": {
    "parent_domain": "A0A5B7_d0",
    "parent_protein": "A0A5B7",
    "score": 0.85,
    "n_residues": 24,
    "residue_ids": ["A_42", "A_43", ...]
  }
}
```

## Graph Construction

Same pipeline as `build_alphafold_graph.py` but on binding site PDBs:
1. Foldseek 3Di encoding of all binding site PDBs
2. Foldseek clustering at TM ≥ 0.7 (pre-clustering for efficiency)
3. All-vs-all search among representatives
4. Union-Find clustering at TM ≥ 0.9
5. ESM-2 encoding of binding site sequences
6. Build PyG graph

Output: `data/graphs/binding_sites_tm09/global_graph.pt`

### Expected Scale
- 47K human domains × ~1.5 pockets/domain = ~70K binding site nodes
- 534K multi-organism domains × ~1.5 = ~800K binding sites (may need to restrict to human initially)

## Anchor Lookup (Inference)

For each IDP:
1. Extract IDRome conformations (same as current)
2. Foldseek search against **binding site DB** (not domain DB)
3. Top-5 anchors per conformation with TM ≥ 0.9
4. Each anchor maps to a binding site node in the graph
5. GATv2 reads out binding site node embedding → cross-attention with drug → pKi

## Training

Same multi-task AffinityGAT (InfoNCE + MSE with curriculum):
- Graph nodes are binding sites
- Anchor cache maps proteins → binding site nodes (not domain nodes)
- Everything else unchanged

## File Structure

```
src/idr_gat/data/p2rank.py                  — P2Rank runner, output parser, pocket PDB extraction
scripts/build_binding_site_graph.py          — Full pipeline: domains → P2Rank → binding site graph
tests/test_p2rank.py                         — Tests for P2Rank integration
```

New files only. Does not modify:
- `scripts/build_alphafold_graph.py` (existing domain graph pipeline)
- `src/idr_gat/model/affinity_gat.py` (model architecture unchanged)
- `scripts/train_affinity_gat.py` (training script unchanged — just point to new graph)

## Output Directories

```
data/processed/binding_site_pdbs/            — Extracted pocket PDBs
data/processed/binding_site_metadata.json    — Pocket → domain → protein mapping
data/graphs/binding_sites_tm09/              — PyG graph + node ranges
```
