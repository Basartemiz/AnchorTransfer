#!/usr/bin/env bash
# Merge results from in-domain and cross-domain runs and print final summary.
set -euo pipefail

cd "$(dirname "$0")/../.."  # project root

echo "=== DrugBAN Paper Replication — Final Results ==="
echo ""

python3 << 'PYEOF'
import pandas as pd
from pathlib import Path

# Merge all result CSVs
dfs = []
for p in Path("results").glob("drugban_paper*.csv"):
    try:
        dfs.append(pd.read_csv(p))
    except Exception:
        pass

if not dfs:
    print("No results found in results/drugban_paper*.csv")
    exit(1)

df = pd.concat(dfs, ignore_index=True)
if "error" in df.columns:
    df = df[df["error"].isna()].drop(columns=["error"])
good = df[df["auroc"].notna()]

# Save merged
merged_path = "results/drugban_paper_all_results.csv"
good.to_csv(merged_path, index=False)

# Print results
print("=" * 95)
print(f"{'Dataset':12s} {'Split':8s} {'Model':28s} {'AUROC':16s} {'AUPRC':16s} {'n':>6s} {'Seeds':>5s}")
print("-" * 95)

for (ds, sp), grp in good.groupby(["dataset", "split"]):
    for md, mgrp in grp.groupby("model"):
        au, ap = mgrp["auroc"], mgrp["auprc"]
        ns = int(mgrp["test_size"].iloc[0]) if pd.notna(mgrp["test_size"].iloc[0]) else "?"
        auroc_str = f"{au.mean():.4f}+-{au.std():.4f}" if len(mgrp) > 1 else f"{au.mean():.4f}"
        auprc_str = f"{ap.mean():.4f}+-{ap.std():.4f}" if len(mgrp) > 1 else f"{ap.mean():.4f}"
        print(f"{ds:12s} {sp:8s} {md:28s} {auroc_str:16s} {auprc_str:16s} {str(ns):>6s} {len(mgrp):>5d}")
    print()

print("=" * 95)
print(f"\nMerged results saved to: {merged_path}")
print(f"Total: {len(good)} successful runs")
PYEOF
