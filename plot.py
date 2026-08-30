"""Figures for the paper, generated from the JSON files in results/.

Run AFTER summarize_results.py, summarize_probe.py and summarize_crossmass.py.
Any figure whose input file is missing is skipped with a message, so this can
be run at any point without failing.

All text is in English and titles are OFF by default: in a paper the caption
carries the explanation, an embedded title duplicates it. Use --titles to get
titles back (handy for slides or for a quick look while iterating).

Sizes are chosen for a two-column IEEEtran layout:
    single column  ~3.4 in wide  (\\includegraphics[width=\\columnwidth]{...})
    double column  ~7.1 in wide  (figure*, width=\\textwidth)

Usage:
    python plot.py
    python plot.py --titles          # with titles, for slides
    python plot.py --format pdf      # vector figures, better for LaTeX
"""

import argparse
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
C_TRAINED = "#4C72B0"
C_RANDOM = "#C44E52"

SINGLE = 3.4   # inches, IEEEtran column width
DOUBLE = 7.1   # inches, IEEEtran text width

ARGS = None    # filled in main()


def style():
    plt.rcParams.update({
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.dpi": 110,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"  (skipped: missing {path})")
        return None


def title(ax, text):
    if ARGS.titles:
        ax.set_title(text)


def save(fig, name):
    os.makedirs(IMAGES, exist_ok=True)
    path = os.path.join(IMAGES, f"{name}.{ARGS.format}")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path}")


# --------------------------------------------------------------------------
# Block 1: policies
# --------------------------------------------------------------------------

def plot_policy_comparison(summary):
    if not summary:
        return
    rows = sorted(summary, key=lambda r: r["target_mean"])

    def short(r):
        m = (r["method"].split("(")[0].strip()
             .replace("Baseline no randomization", "No rand.")
             .replace("Uniform Domain Randomization", "UDR"))
        alg = "LSTM" if "lstm=128" in r["config"] else "FF"
        env = "src" if "source" in r["env"] else "tgt"
        return f"{m}\n{alg}, {env}\n(n={r['n_runs']})"

    labels = [short(r) for r in rows]
    x = np.arange(len(rows))
    w = 0.38

    fig, ax = plt.subplots(figsize=(DOUBLE, 3.0))
    ax.bar(x - w / 2, [r["source_mean"] for r in rows], w,
           yerr=[r["source_std"] for r in rows], capsize=3,
           label="evaluated on source", color="#8C8C8C")
    ax.bar(x + w / 2, [r["target_mean"] for r in rows], w,
           yerr=[r["target_std"] for r in rows], capsize=3,
           label="evaluated on target", color=C_TRAINED)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel("Mean return (50 episodes)")
    title(ax, "Policy comparison; error bars are std across seeds")
    ax.legend(loc="upper left")
    ax.grid(axis="x", visible=False)
    save(fig, "policy_comparison")


def plot_crossmass(cm):
    """Two panels: all three masses scaled together (across seeds), and one
    link at a time (single seed). The second panel is what identifies the foot
    as the only mass that matters for control."""
    if not cm or "all" not in cm:
        return

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.6))

    ax = axes[0]
    for algo, st in [("PPO", dict(color="#8C8C8C", marker="o", ls="-")),
                     ("RecurrentPPO", dict(color=C_TRAINED, marker="s", ls="-"))]:
        d = cm["all"].get(algo)
        if not d:
            continue
        s = np.array(d["scales"])
        m = np.array(d["mean"])
        sd = np.array(d["std"])
        lbl = ("PPO feedforward" if algo == "PPO" else "RecurrentPPO")
        ax.plot(s, m, label=f"{lbl} (n={d['n_runs']})", lw=1.5, ms=4, **st)
        ax.fill_between(s, m - sd, m + sd, color=st["color"], alpha=0.15, lw=0)
    ax.axvspan(0.85, 1.15, color="k", alpha=0.06, lw=0)
    ax.annotate("UDR training range", xy=(1.0, 0.955), xycoords=("data", "axes fraction"),
                fontsize=6, ha="center", va="top", color="0.35")
    ax.set_xlabel("mass scale (thigh, leg and foot together)")
    ax.set_ylabel("Mean return")
    title(ax, "All randomized masses scaled")
    ax.legend(loc="lower left", framealpha=0.9)

    ax = axes[1]
    for link in MASS_NAMES:
        d = (cm.get(link) or {}).get("PPO")
        if not d:
            continue
        ax.plot(d["scales"], d["mean"], marker="o", ms=4, lw=1.5,
                color=COLORS[link], label=link)
    ax.set_xlabel("mass scale (one link at a time)")
    ax.set_ylabel("Mean return")
    title(ax, "PPO feedforward, single seed")
    ax.legend(loc="lower left")

    fig.tight_layout()
    save(fig, "crossmass")


# --------------------------------------------------------------------------
# Block 2: probe
# --------------------------------------------------------------------------

def plot_probe_trained_vs_random(ps):
    """Trained LSTM vs. a reservoir baseline (randomly-initialized LSTM).

    A randomly-initialized recurrent network is already a non-linear
    projection of the observation history, so a lot is decodable from it
    without anything having been learned; this is the baseline decodability
    has to beat to count as evidence of LEARNED system identification.

    With only 3 policy seeds the trained mean is not reliably above or below
    the reservoir band (see the std): read this figure together with
    reward_vs_decodability, which explains why the spread across policy seeds
    is so much larger than across encoder seeds in the first place.
    """
    if not ps:
        return
    tr, rd = ps["trained"], ps["random"]
    if not tr["n_runs"] or not rd["n_runs"]:
        print("  (skipped probe_trained_vs_random: need both trained and control runs)")
        return

    x = np.arange(len(MASS_NAMES))
    w = 0.36
    fig, ax = plt.subplots(figsize=(SINGLE, 2.4))

    for off, block, color, lbl in [
        (-w / 2, tr, C_TRAINED, f"trained LSTM (n={tr['n_runs']} policy seeds)"),
        (+w / 2, rd, C_RANDOM, f"random LSTM (n={rd['n_runs']} encoder seeds)"),
    ]:
        means = [block["per_mass"][m]["mean"] for m in MASS_NAMES]
        stds = [block["per_mass"][m]["std"] for m in MASS_NAMES]
        ax.bar(x + off, means, w, yerr=stds, capsize=3, color=color, label=lbl)
        for xi, mu, sd in zip(x + off, means, stds):
            ax.text(xi, mu + sd + 0.6, f"{mu:.1f}", ha="center", fontsize=6)

    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(MASS_NAMES)
    ax.set_ylabel("MAE reduction vs\nmean-predictor baseline [%]")
    title(ax, "Training does not add decodable mass information")
    ax.legend(loc="upper center", frameon=False)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(top=max(ax.get_ylim()[1], 1) * 1.30)
    save(fig, "probe_trained_vs_random")


def plot_reward_vs_decodability(ps):
    """Decodability tracks how well each policy seed converged, not whether it
    was trained at all. This is the more precise claim than a flat
    'trained vs random' comparison once seed variance turned out to be large:
    the seed with the best transfer reward is the one with the LEAST
    decodable mass information, consistently across all three masses.
    """
    if not ps:
        return
    rc = ps.get("reward_correlation")
    if not rc:
        print("  (skipped reward_vs_decodability: no reward_correlation block, "
              "need >=3 policy seeds with matching results json)")
        return

    rewards = rc["reward_source_to_target"]
    fig, ax = plt.subplots(figsize=(SINGLE, 2.4))
    for m in MASS_NAMES:
        c = rc["per_mass"][m]
        ax.scatter(rewards, c["decodability"], color=COLORS[m], s=28,
                   label=f"{m} (r={c['pearson_r']:+.2f})", zorder=3)
        # simple least-squares line, purely to show the direction (n=3: a
        # trend line here is illustrative, not a fitted model to trust)
        if len(rewards) >= 2:
            z = np.polyfit(rewards, c["decodability"], 1)
            xs = np.linspace(min(rewards), max(rewards), 20)
            ax.plot(xs, np.polyval(z, xs), color=COLORS[m], lw=1.0, alpha=0.5, zorder=2)

    ax.set_xlabel("policy reward, source→target")
    ax.set_ylabel("MAE reduction vs\nmean-predictor baseline [%]")
    title(ax, f"Better-converged seeds encode less mass information (n={rc['n_seeds']})")
    ax.legend(loc="upper right", frameon=False, fontsize=6.5)
    save(fig, "reward_vs_decodability")


def plot_probe_windows_trained_vs_random(ps):
    """The temporal phenomenology (foot identified first, thigh later, then
    decay) side by side for the trained and the random encoder. If the random
    encoder reproduces the shape, the shape is architectural, not learned."""
    if not ps:
        return
    bw = ps.get("by_window") or {}
    tr, rd = bw.get("trained"), bw.get("random")
    if not tr or not rd:
        print("  (skipped probe_windows_trained_vs_random: missing window data)")
        return

    def keys_and_centres(block):
        keys = sorted(block["windows"], key=lambda k: int(k.split("-")[1]))
        return keys, [(int(k.split("-")[0]) + int(k.split("-")[1])) / 2 for k in keys]

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.4), sharey=True)
    for ax, block, name in [(axes[0], tr, "trained LSTM"),
                            (axes[1], rd, "randomly-initialized LSTM")]:
        keys, cx = keys_and_centres(block)
        for m in MASS_NAMES:
            vals = [block["windows"][k][m]["mean"] for k in keys]
            errs = [block["windows"][k][m]["std"] for k in keys]
            ax.errorbar(cx, vals, yerr=errs, marker="o", ms=3.5, lw=1.4,
                        capsize=2, color=COLORS[m], label=m)
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.set_xlabel("timestep in the episode (window centre)")
        ax.set_title(f"{name}  (n={block['n_runs']})", fontsize=8)
    axes[0].set_ylabel("MAE reduction [%]")
    axes[0].legend(loc="upper right", ncol=3, frameon=False)
    fig.tight_layout()
    save(fig, "probe_windows_trained_vs_random")


def plot_probe_improvement(rob):
    """Trained probe alone. Superseded by probe_trained_vs_random for the
    paper, kept because it is the natural first figure when presenting the
    result before the control is introduced."""
    if not rob:
        return
    cv = rob["cross_validation"]
    means = [cv[m]["improv_pct_mean"] for m in MASS_NAMES]
    stds = [cv[m]["improv_pct_std"] for m in MASS_NAMES]

    fig, ax = plt.subplots(figsize=(SINGLE, 2.2))
    ax.bar(MASS_NAMES, means, yerr=stds, capsize=4,
           color=[COLORS[m] for m in MASS_NAMES])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("MAE reduction vs\nmean-predictor baseline [%]")
    title(ax, "Mass information decodable from the hidden state")
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.4, f"{m:.1f}%", ha="center", fontsize=7)
    ax.grid(axis="x", visible=False)
    save(fig, "probe_baseline_improvement")


def plot_probe_early_window(rob):
    if not rob or "early_window" not in rob:
        return
    ew = rob["early_window"]
    fig, ax = plt.subplots(figsize=(SINGLE, 2.3))
    for m in MASS_NAMES:
        buckets = ew["per_mass"][m]["buckets"]
        ax.plot([(b["lo"] + b["hi"]) / 2 for b in buckets],
                [b["improv_pct"] for b in buckets],
                marker="o", ms=3.5, lw=1.4, label=m, color=COLORS[m])
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("timestep in the episode")
    ax.set_ylabel("MAE reduction [%]")
    title(ax, f"First {ew['cutoff_step']:.0f} steps "
              f"({ew['n_episodes_alive_pct']:.0f}% of test episodes alive)")
    ax.legend(ncol=3, frameon=False, loc="upper right")
    save(fig, "probe_early_window")


def plot_probe_controls(ctrl):
    if not ctrl:
        return
    conditions = [
        ("MLP on\nhidden state", "hidden_state"),
        ("Linear\nridge probe", "linear_ridge_probe"),
        ("Control:\ntimestep only", "control_timestep_only"),
        ("Control:\nshuffled labels", "control_shuffled_labels"),
    ]
    conditions = [(lbl, k) for lbl, k in conditions if k in ctrl]

    x = np.arange(len(conditions))
    w = 0.25
    fig, ax = plt.subplots(figsize=(SINGLE, 2.3))
    for i, m in enumerate(MASS_NAMES):
        ax.bar(x + (i - 1) * w, [ctrl[k][m] for _, k in conditions], w,
               label=m, color=COLORS[m])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for lbl, _ in conditions], fontsize=6.5)
    ax.set_ylabel("MAE reduction [%]")
    title(ax, "Signal is not survivorship bias nor label leakage")
    ax.legend(ncol=3, frameon=False, loc="upper right")
    ax.grid(axis="x", visible=False)
    save(fig, "probe_controls")


# --------------------------------------------------------------------------

def main():
    global ARGS
    p = argparse.ArgumentParser()
    p.add_argument("--titles", action="store_true",
                   help="draw titles inside the figures (for slides)")
    p.add_argument("--format", default="png", choices=["png", "pdf"])
    p.add_argument("--results_dir", default=RESULTS)
    ARGS = p.parse_args()
    style()

    print(f"Writing figures to {IMAGES}/ (.{ARGS.format})")
    r = ARGS.results_dir
    summary = load(os.path.join(r, "summary.json"))
    if summary is None:
        print("  hint: run `python summarize_results.py` first")
    probe_sum = load(os.path.join(r, "probe_summary.json"))
    if probe_sum is None:
        print("  hint: run `python summarize_probe.py` first")
    crossmass = load(os.path.join(r, "crossmass_summary.json"))
    if crossmass is None:
        print("  hint: run `python summarize_crossmass.py` first")
    rob = load(os.path.join(r, "probe_robustness_s42.json")) or \
        load(os.path.join(r, "probe_robustness_results.json"))
    ctrl = load(os.path.join(r, "probe_controls_s42.json")) or \
        load(os.path.join(r, "probe_controls_results.json"))

    plot_policy_comparison(summary)
    plot_crossmass(crossmass)
    plot_probe_trained_vs_random(probe_sum)
    plot_reward_vs_decodability(probe_sum)
    plot_probe_windows_trained_vs_random(probe_sum)
    plot_probe_improvement(rob)
    plot_probe_early_window(rob)
    plot_probe_controls(ctrl)
    print("Done.")


if __name__ == "__main__":
    main()
