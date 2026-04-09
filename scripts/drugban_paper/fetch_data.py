"""Download DrugBAN paper datasets from GitHub."""
import os
import urllib.request
from pathlib import Path

REPO_BASE = "https://raw.githubusercontent.com/peizhenbai/DrugBAN/main/datasets"

# All files we need: (dataset, split_dir, filename)
FILES = [
    # BindingDB random
    ("bindingdb", "random", "train.csv"),
    ("bindingdb", "random", "val.csv"),
    ("bindingdb", "random", "test.csv"),
    # BindingDB cluster (cross-domain)
    ("bindingdb", "cluster", "source_train.csv"),
    ("bindingdb", "cluster", "target_train.csv"),
    ("bindingdb", "cluster", "target_test.csv"),
    # BioSNAP random
    ("biosnap", "random", "train.csv"),
    ("biosnap", "random", "val.csv"),
    ("biosnap", "random", "test.csv"),
    # BioSNAP cluster
    ("biosnap", "cluster", "source_train.csv"),
    ("biosnap", "cluster", "target_train.csv"),
    ("biosnap", "cluster", "target_test.csv"),
    # Human random
    ("human", "random", "train.csv"),
    ("human", "random", "val.csv"),
    ("human", "random", "test.csv"),
    # Human cold
    ("human", "cold", "train.csv"),
    ("human", "cold", "val.csv"),
    ("human", "cold", "test.csv"),
]


def fetch_all(data_dir: str = "data/drugban_paper") -> None:
    """Download all DrugBAN paper CSVs if not already present."""
    data_dir = Path(data_dir)
    for dataset, split, fname in FILES:
        out_path = data_dir / dataset / split / fname
        if out_path.exists():
            continue
        url = f"{REPO_BASE}/{dataset}/{split}/{fname}"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {url} -> {out_path}")
        urllib.request.urlretrieve(url, out_path)
    print(f"All DrugBAN paper data ready in {data_dir}")


if __name__ == "__main__":
    fetch_all()
