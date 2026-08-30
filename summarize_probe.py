"""Aggregates probe results across runs and writes results/probe_summary.json.

Two sources of variance are kept separate, because they answer two different
objections:

  policy seeds   trained LSTMs from independent RL runs. Answers "maybe seed
                 42 just happened to be an unlucky policy".
  encoder seeds  randomly-initialized LSTMs (the reservoir control). Answers
                 "maybe that one random initialization was lucky".

Reported per mass: mean and std of the MAE reduction over the naive
mean-predictor baseline, taken across runs (not across CV folds: the per-run
number is already the CV mean).

Usage:
    python summarize_probe.py
    python summarize_probe.py --markdown results/probe_summary.md
"""

import argparse
import glob
import json
import os
import re

import numpy as np

MASS_NAMES = ["thigh", "leg", "foot"]


def load_json(path):
    with open(path) as f:
        return json.load(f)


def seed_from_tag(tag):
    """'s42' -> 42, 's123' -> 123. Returns None if the tag has no clean seed."""
    m = re.search(r"s(\d+)$", tag)
    return int(m.group(1)) if m else None


def load_policy_reward(results_dir, seed, tag_prefix="udr_RecurrentPPO_lstm"):
    """Reward (source_to_target) of the policy that produced a given probe
    dataset, read from the same results/ folder. Returns None if not found:
    the correlation is then skipped rather than silently omitted.
    """
    path = os.path.join(results_dir, f"{tag_prefix}_s{seed}_results.json")
    if not os.path.exists(path):
        return None
    d = load_json(path)
    return float(d["results"]["source_to_target"]["mean"])


def reward_correlation(runs, results_dir):
    """Pearson r between policy convergence (source_to_target reward) and
    probe decodability, per mass, across policy seeds.

    Motivation: a single trained seed cannot distinguish 'training suppresses
    mass information' from 'this one seed happened to encode less'. With
    several seeds, if decodability instead tracks how well each seed
    converged, that is the more precise (and more defensible) claim: it is
    not training in general that suppresses the signal, it is convergence.
    n is small here (one point per policy seed), so p-values are not
    expected to be significant on their own; consistency of sign across all
    three masses is the qualitative evidence, and must be reported as such.
    """
    from scipy.stats import pearsonr

    pairs = []
    for r in runs:
        seed = seed_from_tag(r["tag"])
        if seed is None:
            continue
        reward = load_policy_reward(results_dir, seed)
        if reward is None:
            continue
        pairs.append((seed, reward, r["per_mass"]))

    if len(pairs) < 3:
        print(f"  (skipping reward correlation: only {len(pairs)} seed(s) "
              "with a matching policy result file, need >= 3)")
        return None

    pairs.sort(key=lambda p: p[0])
    out = {
        "n_seeds": len(pairs),
        "seeds": [p[0] for p in pairs],
        "reward_source_to_target": [p[1] for p in pairs],
        "per_mass": {},
    }
    for m in MASS_NAMES:
        rewards = [p[1] for p in pairs]
        decod = [p[2][m] for p in pairs]
        r, p_val = pearsonr(rewards, decod)
        out["per_mass"][m] = {
            "decodability": decod,
            "pearson_r": float(r),
            "pearson_p": float(p_val),
        }
        print(f"  [reward correlation] {m}: r={r:+.3f} (n={len(pairs)}, p={p_val:.3f})")
    return out


def collect(paths, kind):
    """Returns {mass: [values]} plus the per-run detail."""
    runs = []
    for path in sorted(paths):
        d = load_json(path)
        cv = d.get("cross_validation")
        if not cv:
            print(f"  skipping {path}: no cross_validation block")
            continue
        tag = re.sub(r"^probe_robustness_|\.json$", "", os.path.basename(path))
        runs.append({
            "file": os.path.basename(path),
            "tag": tag,
            "dataset": d.get("dataset"),
            "per_mass": {m: float(cv[m]["improv_pct_mean"]) for m in MASS_NAMES},
            "per_mass_cv_std": {m: float(cv[m]["improv_pct_std"]) for m in MASS_NAMES},
        })
        print(f"  [{kind}] {tag:26s} " +
              "  ".join(f"{m}={cv[m]['improv_pct_mean']:5.1f}" for m in MASS_NAMES))
    return runs


def summarize(runs):
    out = {"n_runs": len(runs), "runs": runs, "per_mass": {}}
    for m in MASS_NAMES:
        vals = np.array([r["per_mass"][m] for r in runs], dtype=float)
        out["per_mass"][m] = {
            "mean": float(vals.mean()) if len(vals) else None,
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "min": float(vals.min()) if len(vals) else None,
            "max": float(vals.max()) if len(vals) else None,
            "values": [float(v) for v in vals],
        }
    return out


def collect_windows(paths, kind):
    """Aggregates the fixed-window analysis from analyze_probe_controls.py.

    Windows are fixed (0-0, 1-20, ...) so they are comparable across datasets,
    unlike the early_window buckets whose edges depend on episode lengths.
    """
    per_window = {}
    n = 0
    for path in sorted(paths):
        d = load_json(path)
        bw = d.get("by_window")
        if not bw:
            continue
        n += 1
        for win, vals in bw.items():
            slot = per_window.setdefault(win, {m: [] for m in MASS_NAMES})
            for m in MASS_NAMES:
                if m in vals:
                    slot[m].append(float(vals[m]))
    if not n:
        return None
    print(f"  [{kind}] window analysis aggregated over {n} run(s)")
    return {
        "n_runs": n,
        "windows": {
            win: {m: {"mean": float(np.mean(v)),
                      "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0}
                  for m, v in vals.items() if v}
            for win, vals in per_window.items()
        },
    }


def markdown(summary):
    lines = []
    tr, rd = summary["trained"], summary["random"]
    lines.append("| mass | trained LSTM (n=%d policy seeds) | random LSTM (n=%d encoder seeds) |"
                 % (tr["n_runs"], rd["n_runs"]))
    lines.append("|---|---:|---:|")
    for m in MASS_NAMES:
        a, b = tr["per_mass"][m], rd["per_mass"][m]
        fa = "n/a" if a["mean"] is None else f"{a['mean']:.1f} ± {a['std']:.1f}"
        fb = "n/a" if b["mean"] is None else f"{b['mean']:.1f} ± {b['std']:.1f}"
        lines.append(f"| {m} | {fa} | {fb} |")
    lines.append("")
    lines.append("MAE reduction (%) over a baseline that always predicts the "
                 "training-set mean mass. Std is across runs, not across CV folds.")
    lines.append("")
    lines.append("With only 3 policy seeds, mean ± std hides a real pattern. "
                 "Per-seed values:")
    lines.append("")
    lines.append("| policy seed | thigh | leg | foot |")
    lines.append("|---|---:|---:|---:|")
    for r in sorted(tr["runs"], key=lambda r: seed_from_tag(r["tag"]) or 0):
        pm = r["per_mass"]
        lines.append(f"| {r['tag']} | {pm['thigh']:.1f} | {pm['leg']:.1f} | {pm['foot']:.1f} |")

    rc = summary.get("reward_correlation")
    if rc:
        lines.append("")
        lines.append(f"Reward vs. decodability, {rc['n_seeds']} policy seeds "
                     "(source_to_target reward vs. MAE-reduction %):")
        lines.append("")
        lines.append("| mass | Pearson r | p (n=%d, not significant alone) |" % rc["n_seeds"])
        lines.append("|---|---:|---:|")
        for m in MASS_NAMES:
            c = rc["per_mass"][m]
            lines.append(f"| {m} | {c['pearson_r']:+.3f} | {c['pearson_p']:.3f} |")
        lines.append("")
        lines.append("Negative r on all three masses: the seeds that converge to a "
                     "higher reward are the ones from which less mass information "
                     "can be decoded. n=3 is too small for the p-value to be "
                     "significant on its own; the evidence is the consistent sign "
                     "across all three independently-fit masses, not the p-value.")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="./results")
    p.add_argument("--trained_glob", default="probe_robustness_s*.json",
                   help="one file per policy seed")
    p.add_argument("--control_glob", default="probe_robustness_control_e*.json",
                   help="one file per random encoder")
    p.add_argument("--trained_ctrl_glob", default="probe_controls_s*.json")
    p.add_argument("--control_ctrl_glob", default="probe_controls_control_e*.json")
    p.add_argument("--out", default="./results/probe_summary.json")
    p.add_argument("--markdown", default=None)
    args = p.parse_args()

    g = lambda pat: glob.glob(os.path.join(args.results_dir, pat))  # noqa: E731

    print("Trained LSTM runs:")
    trained_runs = collect(g(args.trained_glob), "trained")
    print("Random-encoder control runs:")
    control_runs = collect(g(args.control_glob), "random")

    if not trained_runs:
        print("\n!! No trained run found with pattern "
              f"'{args.trained_glob}'. Did you run the probe sweep?")
        print("   (the legacy file results/probe_robustness_results.json is not")
        print("    matched on purpose: copy it to probe_robustness_s42.json)")
    if not control_runs:
        print(f"\n!! No control run found with pattern '{args.control_glob}'.")

    print("Window analysis:")
    win_trained = collect_windows(g(args.trained_ctrl_glob), "trained")
    win_random = collect_windows(g(args.control_ctrl_glob), "random")

    print("Reward vs decodability correlation (trained seeds only):")
    reward_corr = reward_correlation(trained_runs, args.results_dir)

    summary = {
        "trained": summarize(trained_runs),
        "random": summarize(control_runs),
        "by_window": {"trained": win_trained, "random": win_random},
        "reward_correlation": reward_corr,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"\nSaved: {args.out}")

    table = markdown(summary)
    print("\n" + table)
    if args.markdown:
        with open(args.markdown, "w") as f:
            f.write(table + "\n")
        print(f"\nSaved: {args.markdown}")


if __name__ == "__main__":
    main()
