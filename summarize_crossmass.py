"""Aggregates the cross-mass evaluations into results/crossmass_summary.json.

Cross-mass asks a specific question: UDR randomizes thigh/leg/foot, and those
are exactly the masses that are IDENTICAL in source and target. So the failure
of UDR on the target could be explained purely by the torso mismatch. If that
were the whole story, then on environments where thigh/leg/foot actually vary,
memory should pay off. This aggregates those evaluations so the claim can be
checked across seeds instead of on a single run.

Input files are produced by eval_cross_mass.py and named
    results/crossmass_{ALGO}_{WHICH}_{TAG}.json

Usage:
    python summarize_crossmass.py
    python summarize_crossmass.py --markdown results/crossmass_summary.md
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

ALGO_LABEL = {"PPO": "PPO feedforward", "RecurrentPPO": "RecurrentPPO"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="./results")
    p.add_argument("--out", default="./results/crossmass_summary.json")
    p.add_argument("--markdown", default=None)
    args = p.parse_args()

    paths = sorted(glob.glob(os.path.join(args.results_dir, "crossmass_*.json")))
    if not paths:
        print(f"No crossmass_*.json in {args.results_dir}")
        return

    # (which, algorithm) -> scale -> [per-run means]
    agg = defaultdict(lambda: defaultdict(list))
    files = defaultdict(list)

    for path in paths:
        if os.path.basename(path) == "crossmass_summary.json":
            continue
        with open(path) as f:
            d = json.load(f)
        key = (d.get("which", "all"), d.get("algorithm", "?"))
        files[key].append(os.path.basename(path))
        for row in d["results"]:
            agg[key][float(row["scale"])].append(float(row["mean"]))

    summary = {}
    for (which, algo), by_scale in sorted(agg.items()):
        scales = sorted(by_scale)
        summary.setdefault(which, {})[algo] = {
            "n_runs": max(len(by_scale[s]) for s in scales),
            "files": sorted(files[(which, algo)]),
            "scales": scales,
            "mean": [float(np.mean(by_scale[s])) for s in scales],
            "std": [float(np.std(by_scale[s], ddof=1)) if len(by_scale[s]) > 1 else 0.0
                    for s in scales],
        }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"Saved: {args.out}\n")

    lines = []
    for which, by_algo in summary.items():
        algos = sorted(by_algo)
        scales = by_algo[algos[0]]["scales"]
        n = {a: by_algo[a]["n_runs"] for a in algos}
        lines.append(f"### scaling `{which}`  "
                     + ", ".join(f"{ALGO_LABEL.get(a, a)}: n={n[a]}" for a in algos))
        lines.append("")
        lines.append("| scale | " + " | ".join(ALGO_LABEL.get(a, a) for a in algos) + " |")
        lines.append("|---:|" + "---:|" * len(algos))
        for i, s in enumerate(scales):
            cells = []
            for a in algos:
                m = by_algo[a]["mean"][i]
                sd = by_algo[a]["std"][i]
                cells.append(f"{m:.0f} ± {sd:.0f}" if n[a] > 1 else f"{m:.0f}")
            lines.append(f"| {s:.2f} | " + " | ".join(cells) + " |")
        lines.append("")

    table = "\n".join(lines)
    print(table)
    if args.markdown:
        with open(args.markdown, "w") as f:
            f.write(table + "\n")
        print(f"Saved: {args.markdown}")


if __name__ == "__main__":
    main()
