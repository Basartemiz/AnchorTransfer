"""Fix eval_bdb_to_glass_prot_only.py: skip drug filter, fix CoNCISE config, add NaN handling."""
import shutil

src = "scripts/eval_bdb_to_glass.py"
dst = "scripts/eval_bdb_to_glass_prot_only.py"
shutil.copy2(src, dst)

with open(dst) as f:
    code = f.read()

# 1. Skip drug filter
code = code.replace(
    "glass_filt = glass[~glass.canon_smiles.isin(overlap_drugs)].copy()",
    "glass_filt = glass.copy()  # No drug filtering",
)

# 2. Fix CoNCISE model config
code = code.replace("drug_dim=proj_dim, proj_dim=proj_dim", "drug_dim=128, proj_dim=proj_dim")
code = code.replace('activation="gelu"', 'activation="tanh"')
code = code.replace(
    "nn.Linear(fused_dim, 512), nn.ReLU(), nn.Dropout(0.2),",
    "nn.Linear(fused_dim, 256), nn.ReLU(),",
)
code = code.replace("nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.2),", "")
code = code.replace("nn.Linear(128, 1),", "nn.Linear(256, 1),")
code = code.replace("models/concise_bdb/best_model.pt", "models/concise_bdb_fixed/best_model.pt")

# 3. Add NaN handling to auroc_safe
code = code.replace(
    "return float(roc_auc_score(binder[mask].astype(int), preds[mask]))",
    "nm = ~np.isnan(preds[mask])\n"
    "    if nm.sum() < 2: return float('nan')\n"
    "    return float(roc_auc_score(binder[mask][nm].astype(int), preds[mask][nm]))",
)

# 4. Save output to different file
code = code.replace(
    "results/bdb_to_glass_predictions.csv",
    "results/bdb_to_glass_prot_only_predictions.csv",
)

with open(dst, "w") as f:
    f.write(code)

print(f"Fixed {dst}")
