"""Learning curves read from models/*/evaluations.npz.

Point of this figure: the gap between PPO feedforward and RecurrentPPO is not
end-of-training noise. Feedforward saturates early and stays flat, the
recurrent policy oscillates and has not converged at 1M steps. This is the
answer to "did you try training longer?".

The shaded band is min/max across seeds, not a standard deviation: with 6 runs
it shows the actual envelope instead of implying a distribution we did not
verify.

Output goes to images/ so it sits next to the other paper figures.
"""

import argparse
import glob
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_group(pattern, models_dir):
    """Curves from every matching folder, re-aligned on the shortest timestep
    grid (runs can have different lengths)."""
    curves, ts = [], None
    for folder in sorted(glob.glob(os.path.join(models_dir, pattern, ""))):
        f = os.path.join(folder, "evaluations.npz")
        if not os.path.exists(f):
            continue
        d = np.load(f)
        r = d["results"].mean(axis=1)
        t = d["timesteps"]
        if ts is None or len(t) < len(ts):
            ts = t
        curves.append((t, r))
    if not curves:
        return None, None
    n = len(ts)
    return ts, np.array([r[:n] for _, r in curves])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models_dir", default="./models")
    p.add_argument("--out", default="./images/learning_curves.png")
    p.add_argument("--groups", nargs="*", default=[
        "udr_ff_s*:PPO feedforward + UDR",
        "udr_lstm_s*:RecurrentPPO + UDR",
        "baseline_ff_source_s*:PPO feedforward, no rand.",
        "baseline_source_s*:RecurrentPPO, no rand.",
    ], help="pattern:label")
    p.add_argument("--smooth", type=int, default=3,
                   help="moving average over evaluations (1 = none)")
    p.add_argument("--titles", action="store_true",
                   help="draw a title inside the figure (for slides)")
    args = p.parse_args()

    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    plt.figure(figsize=(3.4, 2.4))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, spec in enumerate(args.groups):
        pattern, _, label = spec.partition(":")
        ts, curves = load_group(pattern, args.models_dir)
        if curves is None:
            print(f"  no run matches {pattern}, skipping")
            continue

        mean = curves.mean(axis=0)
        if args.smooth > 1:
            k = np.ones(args.smooth) / args.smooth
            mean = np.convolve(mean, k, mode="same")
        c = colors[i % len(colors)]
        plt.plot(ts, mean, color=c, lw=1.4,
                 label=f"{label or pattern} (n={len(curves)})")
        if len(curves) > 1:
            plt.fill_between(ts, curves.min(axis=0), curves.max(axis=0),
                             color=c, alpha=0.15, lw=0)
        print(f"  {pattern:26s} n={len(curves)}  final={curves[:, -1].mean():.0f}")

    plt.xlabel("training timesteps")
    plt.ylabel("mean return (training environment)")
    if args.titles:
        plt.title("Learning curves: feedforward vs recurrent")
    plt.legend(loc="lower right", frameon=False)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()