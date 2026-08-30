"""Grafici per il report, letti dai json in results/.

Va lanciato dopo summarize_results.py. Se un file manca il grafico viene
saltato.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

IMAGES = "./images"
RESULTS = "./results"
MASS_NAMES = ["thigh", "leg", "foot"]
COLORS = {"thigh": "#4C72B0", "leg": "#DD8452", "foot": "#55A868"}


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"  (salto: manca {path})")
        return None


def save(fig, name):
    os.makedirs(IMAGES, exist_ok=True)
    path = os.path.join(IMAGES, name)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  salvato: {path}")


def plot_policy_comparison(summary):

    if not summary:
        return
    rows = sorted(summary, key=lambda r: r["target_mean"])
    labels = [f"{r['method'].split('(')[0].strip()}\n{r['config'][:28]}\n(n={r['n_runs']})"
              for r in rows]
    tgt = [r["target_mean"] for r in rows]
    tgt_e = [r["target_std"] for r in rows]
    src = [r["source_mean"] for r in rows]
    src_e = [r["source_std"] for r in rows]

    x = np.arange(len(rows))
    w = 0.38
    fig, ax = plt.subplots(figsize=(max(9, 2.0 * len(rows)), 5.5))
    ax.bar(x - w / 2, src, w, yerr=src_e, capsize=4, label="→ source", color="#8C8C8C")
    ax.bar(x + w / 2, tgt, w, yerr=tgt_e, capsize=4, label="→ target", color="#4C72B0")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Reward medio (50 episodi)")
    ax.set_title("Confronto tra policy: barre di errore = std tra seed")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    save(fig, "policy_comparison.png")


def plot_probe_improvement(rob):

    if not rob:
        return
    cv = rob["cross_validation"]
    means = [cv[m]["improv_pct_mean"] for m in MASS_NAMES]
    stds = [cv[m]["improv_pct_std"] for m in MASS_NAMES]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(MASS_NAMES, means, yerr=stds, capsize=6,
           color=[COLORS[m] for m in MASS_NAMES])
    ax.axhline(0, color="black", lw=1)
    ax.set_ylabel("Riduzione del MAE vs baseline naive [%]")
    ax.set_title("Informazione sulle masse decodificabile dallo hidden state\n"
                 "(cross-validation a 5 fold sugli episodi)")
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.8, f"{m:.1f}%", ha="center", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    save(fig, "probe_baseline_improvement.png")


def plot_probe_early_window(rob):

    if not rob or "early_window" not in rob:
        return
    ew = rob["early_window"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in MASS_NAMES:
        buckets = ew["per_mass"][m]["buckets"]
        centers = [(b["lo"] + b["hi"]) / 2 for b in buckets]
        vals = [b["improv_pct"] for b in buckets]
        ax.plot(centers, vals, marker="o", label=m, color=COLORS[m])
    ax.axhline(0, color="black", lw=1, ls="--")
    ax.set_xlabel("Timestep nell'episodio")
    ax.set_ylabel("Riduzione del MAE vs baseline [%]")
    ax.set_title(f"Identificazione nei primi {ew['cutoff_step']:.0f} step\n"
                 f"({ew['n_episodes_alive_pct']:.0f}% degli episodi di test ancora vivi)")
    ax.legend()
    ax.grid(alpha=0.3)
    save(fig, "probe_early_window.png")


def plot_probe_controls(ctrl):

    if not ctrl:
        return
    conditions = [
        ("MLP su hidden state", "hidden_state"),
        ("Ridge lineare", "linear_ridge_probe"),
        ("Controllo: solo timestep", "control_timestep_only"),
        ("Controllo: label mescolate", "control_shuffled_labels"),
    ]
    conditions = [(lbl, k) for lbl, k in conditions if k in ctrl]

    x = np.arange(len(conditions))
    w = 0.25
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for i, m in enumerate(MASS_NAMES):
        vals = [ctrl[k][m] for _, k in conditions]
        ax.bar(x + (i - 1) * w, vals, w, label=m, color=COLORS[m])
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for lbl, _ in conditions], fontsize=8)
    ax.set_ylabel("Riduzione del MAE vs baseline [%]")
    ax.set_title("Il segnale non e' spiegato da survivorship bias ne' da leakage")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    save(fig, "probe_controls.png")


def plot_probe_time_windows(ctrl):

    if not ctrl or "by_window" not in ctrl:
        return
    wins = ctrl["by_window"]
    keys = list(wins.keys())
    centers = [(int(k.split("-")[0]) + int(k.split("-")[1])) / 2 for k in keys]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for m in MASS_NAMES:
        ax.plot(centers, [wins[k][m] for k in keys], marker="o", label=m, color=COLORS[m])
    ax.axhline(0, color="black", lw=1, ls="--")
    ax.set_xlabel("Timestep (centro della finestra)")
    ax.set_ylabel("Riduzione del MAE vs baseline [%]")
    ax.set_title("Gerarchia di osservabilita': quando ogni massa diventa identificabile")
    ax.legend()
    ax.grid(alpha=0.3)
    save(fig, "probe_time_windows.png")


def main():
    print("Genero i grafici in ./images")
    summary = load(os.path.join(RESULTS, "summary.json"))
    if summary is None:
        print("  suggerimento: esegui prima `python summarize_results.py`")
    rob = load(os.path.join(RESULTS, "probe_robustness_results.json"))
    ctrl = load(os.path.join(RESULTS, "probe_controls_results.json"))

    plot_policy_comparison(summary)
    plot_probe_improvement(rob)
    plot_probe_early_window(rob)
    plot_probe_controls(ctrl)
    plot_probe_time_windows(ctrl)
    print("Fatto.")


if __name__ == "__main__":
    main()
