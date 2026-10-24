"""Aggregate repeated experiment logs into mean +/- std with a paired t-test.

Expects logs named by seed_run.py's convention, where the replicate id is
either a training seed or a split seed:
    {exp}_{ds}_s{seed}.log            -> Test, Oracle sections
    {exp}_{ds}_s{seed}_davis.log      -> Davis-OOD, Davis-oracle sections
    {exp}_{ds}_split{seed}.log        -> same, one protein partition per file

Both models are trained inside one run, so the anchor-vs-baseline difference is
paired and the t is computed on per-replicate differences. The model whose name
contains "anchor" is treated as the anchor arm.

Point one invocation at a directory of seed replicates (split held fixed) and
another at a directory of split replicates (seed held fixed); mixing the two in
one directory would count the shared run twice.

Usage:
    python experiments/summarize_seeds.py <log_dir> [output.json]
"""
from __future__ import annotations

import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

METRIC_RE = re.compile(
    r"^\s{2}(?P<model>[A-Za-z0-9_. \-]+?):\s+macro_CI=(?P<ci>[-\d.]+|nan)\s+"
    r"macro_RMSE=(?P<rmse>[-\d.]+|nan)\s+macro_AUROC=(?P<auroc>[-\d.]+|nan)"
)
NAME_RE = re.compile(
    r"^(?P<exp>experiment\d)_(?P<ds>dtc|bdb)_(?:s|split)(?P<seed>\d+)(?P<davis>_davis)?$"
)

# Longest label first: "DAVIS-OOD — ORACLE ANCHORS" also contains "DAVIS-OOD".
SECTIONS = [
    ("DAVIS-OOD — ORACLE ANCHORS", "Davis-oracle"),
    ("DAVIS-OOD", "Davis-OOD"),
    ("TEST SET — ORACLE ANCHORS", "Oracle"),
    ("TEST SET", "Test"),
]


def section_of(line: str) -> str | None:
    for needle, name in SECTIONS:
        if needle in line:
            return name
    return None


def mean_std(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    mu = sum(xs) / n
    if n < 2:
        return mu, 0.0
    return mu, math.sqrt(sum((x - mu) ** 2 for x in xs) / (n - 1))


def collect(log_dir: Path) -> dict[tuple, dict[int, float]]:
    acc: dict[tuple, dict[int, float]] = defaultdict(dict)
    for log in sorted(log_dir.glob("*.log")):
        m = NAME_RE.match(log.stem)
        if not m:
            continue
        exp, ds, seed = m.group("exp"), m.group("ds"), int(m.group("seed"))
        current = None
        for line in log.read_text(errors="replace").splitlines():
            sec = section_of(line)
            if sec:
                current = sec
                continue
            mm = METRIC_RE.match(line)
            if mm and current:
                model = mm.group("model").strip()
                for metric in ("ci", "rmse", "auroc"):
                    if mm.group(metric) != "nan":
                        acc[(exp, ds, current, metric, model)][seed] = float(mm.group(metric))
    return acc


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    log_dir = Path(sys.argv[1])
    acc = collect(log_dir)

    groups: dict[tuple, list[str]] = defaultdict(list)
    for (exp, ds, section, metric, model) in acc:
        key = (exp, ds, section, metric)
        if model not in groups[key]:
            groups[key].append(model)

    print(f"{'exp':<12} {'ds':<4} {'section':<13} {'metric':<6} "
          f"{'baseline':>17} {'anchor':>17} {'delta':>9} {'t':>7} {'n':>3}")
    print("-" * 100)

    out = {}
    for key in sorted(groups):
        exp, ds, section, metric = key
        models = groups[key]
        anchor = next((m for m in models if "anchor" in m.lower()), None)
        base = next((m for m in models if m != anchor), None)
        if not anchor or not base:
            continue
        a_by, b_by = acc[(*key, anchor)], acc[(*key, base)]
        seeds = sorted(set(a_by) & set(b_by))
        if not seeds:
            continue
        a = [a_by[s] for s in seeds]
        b = [b_by[s] for s in seeds]
        a_mu, a_sd = mean_std(a)
        b_mu, b_sd = mean_std(b)
        d_mu, d_sd = mean_std([x - y for x, y in zip(a, b)])
        t = (d_mu / (d_sd / math.sqrt(len(seeds)))) if len(seeds) > 1 and d_sd > 0 else float("nan")

        print(f"{exp:<12} {ds:<4} {section:<13} {metric:<6} "
              f"{b_mu:>9.4f}±{b_sd:<7.4f} {a_mu:>9.4f}±{a_sd:<7.4f} "
              f"{d_mu:>+9.4f} {t:>7.2f} {len(seeds):>3}")
        out["|".join(key)] = {
            "baseline_model": base, "anchor_model": anchor, "seeds": seeds,
            "baseline_mean": b_mu, "baseline_std": b_sd,
            "anchor_mean": a_mu, "anchor_std": a_sd, "delta": d_mu, "t": t,
        }

    if len(sys.argv) > 2:
        dest = Path(sys.argv[2])
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
