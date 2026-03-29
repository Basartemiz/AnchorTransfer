# DrugTargetCommons Dataset + Disorder Entropy Evaluation

**Date:** 2026-03-29
**Status:** Approved
**Branch:** alphafold

## Problem

1. Training data is limited to 176K interactions from BindingDB benchmark (203 proteins). DrugTargetCommons aggregates BindingDB + ChEMBL + literature into ~3.5M interactions, providing far more training signal for the multi-task InfoNCE + MSE model.

2. Not all IDPs are equally disordered. Evaluation should stratify IDPs by a computable disorder intensity score to understand where the anchor transfer approach works best (conformational selection vs induced fit regime).

## Part 1: DrugTargetCommons Data Pipeline

### Data Source
DrugTargetCommons bulk download (TSV format) from drugtargetcommons.fimm.fi.

### Filtering
- Keep only Ki or Kd measurement types (exclude IC50/EC50 — different assay semantics)
- Require valid UniProt ID (SwissProt accession)
- Require valid SMILES string
- Convert to pKi: `pKi = 9.0 - log10(Ki_nM)` where Ki is in nanomolar
- Clip pKi to [3.0, 12.0] range
- Drop duplicates: if multiple measurements exist for the same (UniProt, SMILES) pair, take the median pKi

### Train/Test Split
- **Test set**: the existing 203-protein benchmark (held out, never trained on)
- **Training set**: all DTC interactions where the UniProt ID is NOT in the 203 benchmark proteins
- This ensures clean out-of-distribution evaluation

### Output Format
`data/processed/dtc_training_interactions.csv`:
```
uniprot_id,ligand_smiles,pki
P12345,CCO,7.5
Q67890,c1ccccc1,5.2
```

Compatible with existing `prepare_training_data()` and `ProteinBatchSampler` — just load the CSV instead of `benchmark_affinity.csv`.

### Expected Scale
- DTC has ~3.5M raw interactions
- After Ki/Kd filter: ~1-2M
- After dedup + valid SMILES/UniProt: ~500K-1M
- After removing benchmark proteins: ~400K-900K training pairs
- Significant increase over 176K current benchmark

## Part 2: Disorder Entropy Score

### Computation

For each IDP, compute a disorder intensity score in [0, 1]:

```python
disorder_score = 0.5 * mean_3di_entropy + 0.5 * (1.0 - mean_plddt)
```

**3Di token entropy**:
- Take the protein's IDRome conformations (multiple PDB frames)
- Encode each conformation with Foldseek 3Di (20-letter structural alphabet)
- For each residue position, compute Shannon entropy across conformations: `H = -Σ p(t) log2 p(t)` where p(t) is the frequency of 3Di token t at that position
- Normalize by log2(20) to get [0, 1] range
- Take the mean across all residue positions

**Mean pLDDT**:
- From AlphaFold predicted structure (available via AlphaFold DB API or pre-computed)
- Mean per-residue pLDDT, normalized to [0, 1] by dividing by 100
- Inverted: `(1 - mean_plddt)` so higher = more disordered

### Binning

Split the 170 benchmark IDPs into 4 quartile bins based on disorder_score. Report evaluation metrics per bin:

| Quartile | Disorder Score Range | n | seq_CI | anc_CI | dCI | seq_MSE | anc_MSE | dMSE% |
|---|---|---|---|---|---|---|---|---|

### Interpretation
- **Q1 (lowest disorder)**: IDPs with more residual structure → conformational selection regime → anchor transfer should work well (free ensemble contains binding-competent states)
- **Q4 (highest disorder)**: fully disordered IDPs → induced fit regime → anchor transfer may be less effective (bound conformation not in free ensemble)

## File Structure

```
src/idr_gat/data/dtc_loader.py           — Download, filter, deduplicate DTC data
src/idr_gat/evaluation/disorder_score.py  — Compute 3Di entropy + pLDDT disorder score
scripts/evaluate_anchor_dta.py            — Add --disorder-quartiles flag for binned reporting
tests/test_affinity_gat_multitask.py      — Tests for DTC loader + disorder score
```

## Integration with Existing Pipeline

DTC data feeds into `train_affinity_gat.py` via:
1. `dtc_loader.py` downloads and produces the CSV
2. `prepare_training_data()` reads the CSV (same format as benchmark)
3. `ProteinBatchSampler` handles the pos/neg/mid splitting
4. No changes to model architecture

Disorder score feeds into `evaluate_anchor_dta.py` via:
1. `disorder_score.py` computes scores for benchmark proteins
2. Eval script bins proteins by quartile and reports per-bin metrics
3. Scores saved alongside results for further analysis
