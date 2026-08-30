"""Curve di apprendimento da models/*/evaluations.npz.

Serve a mostrare che il divario tra PPO feedforward e RecurrentPPO non e' una
questione di rumore finale: il feedforward satura presto e resta piatto, il
ricorrente oscilla e a 1M non ha convergito. E' l'argomento da opporre alla
domanda "avete provato ad addestrare piu' a lungo".

La banda e' min/max tra i seed, non una deviazione standard: con 6 run mostra
l'inviluppo effettivo invece di suggerire una distribuzione che non abbiamo
verificato.
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
    """Curve di tutte le cartelle che matchano, riallineate sulla griglia
    di timesteps piu' corta (i run possono avere lunghezze diverse)."""
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
    p.add_argument("--out", default="./results/learning_curves.png")
    p.add_argument("--groups", nargs="*", default=[
        "udr_ff_s*:PPO feedforward + UDR",
        "udr_lstm_s*:RecurrentPPO + UDR",
        "baseline_ff_source_s*:PPO feedforward, no rand.",
        "baseline_source_s*:RecurrentPPO, no rand.",
    ], help="pattern:etichetta")
    p.add_argument("--smooth", type=int, default=3,
                   help="media mobile sulle valutazioni (1 = nessuna)")
    args = p.parse_args()

    plt.figure(figsize=(7.5, 4.5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, spec in enumerate(args.groups):
        pattern, _, label = spec.partition(":")
        ts, curves = load_group(pattern, args.models_dir)
        if curves is None:
            print(f"  nessun run per {pattern}, salto")
            continue

        mean = curves.mean(axis=0)
        if args.smooth > 1:
            k = np.ones(args.smooth) / args.smooth
            mean = np.convolve(mean, k, mode="same")
        c = colors[i % len(colors)]
        plt.plot(ts, mean, color=c, lw=1.8,
                 label=f"{label or pattern} (n={len(curves)})")
        if len(curves) > 1:
            plt.fill_between(ts, curves.min(axis=0), curves.max(axis=0),
                             color=c, alpha=0.15, lw=0)
        print(f"  {pattern:26s} n={len(curves)}  finale={curves[:, -1].mean():.0f}")

    plt.xlabel("timesteps di training")
    plt.ylabel("return medio (eval sull'ambiente di training)")
    plt.title("Curve di apprendimento: feedforward vs ricorrente")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=150)
    print(f"\nSalvato in {args.out}")


if __name__ == "__main__":
    main()