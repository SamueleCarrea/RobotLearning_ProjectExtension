"""Analisi di robustezza del probe.

Due cose: cross-validation a k fold sugli episodi (il risultato non deve
dipendere dallo split), e analisi dei primi step dell'episodio, dove quasi
tutti gli episodi sono ancora vivi e quindi non c'e' survivorship bias.
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, TensorDataset

from probe_model import ProbeMLP

MASS_NAMES = ["thigh", "leg", "foot"]


def print_metadata(data):
    keys = [k for k in data.files if k.startswith("meta_")]
    if not keys:
        print("(dataset senza metadata, non si sa da che policy viene)")
        return
    print("Metadata del dataset:")
    for k in sorted(keys):
        print(f"  {k[5:]:18s} = {data[k]}")


def make_episode_folds(episodes, n_folds, seed):
    # fold sugli episodi, non sulle righe, altrimenti c'e' leakage
    uniq = np.unique(episodes)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    return np.array_split(uniq, n_folds)


def train_one_fold(X_train, y_train, hidden_dim, epochs, lr, batch_size, device):
    mu, sd = y_train.mean(0), y_train.std(0) + 1e-8
    model = ProbeMLP(X_train.shape[1], hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor((y_train - mu) / sd, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
    return model, mu, sd


def predict_denorm(model, X, mu, sd, device):
    model.eval()
    with torch.no_grad():
        pred_n = model(torch.tensor(X, dtype=torch.float32, device=device)).cpu().numpy()
    return pred_n * sd + mu


def cross_validate(data, n_folds, epochs, hidden_dim, lr, batch_size, device, seed):
    X, y, t, episode = data["X"], data["y"], data["t"], data["episode"]
    folds = make_episode_folds(episode, n_folds, seed)

    improvs, corrs = [], []
    for k in range(n_folds):
        is_test = np.isin(episode, folds[k])
        is_train = ~is_test

        model, mu, sd = train_one_fold(X[is_train], y[is_train], hidden_dim,
                                       epochs, lr, batch_size, device)
        pred = predict_denorm(model, X[is_test], mu, sd, device)
        err = np.abs(pred - y[is_test])

        baseline_mae = np.abs(y[is_train].mean(0, keepdims=True) - y[is_test]).mean(0)
        improv = 100 * (1 - err.mean(0) / baseline_mae)
        corr = np.array([pearsonr(t[is_test], err[:, i])[0] for i in range(3)])

        improvs.append(improv)
        corrs.append(corr)
        print(f"  fold {k + 1}/{n_folds}: improv%={np.round(improv, 1)}  "
              f"pearson_r={np.round(corr, 3)}")

    improvs, corrs = np.array(improvs), np.array(corrs)
    print("\nRiepilogo cross-validation (media +/- std sui fold):")
    for i, name in enumerate(MASS_NAMES):
        print(f"  {name}: improv% = {improvs[:, i].mean():.1f} +/- {improvs[:, i].std():.1f}"
              f"   |   pearson_r = {corrs[:, i].mean():.3f} +/- {corrs[:, i].std():.3f}")
    return improvs, corrs


def early_window_analysis(data, model, mu, sd, device, n_buckets, survival_frac,
                          is_test):
    # bucket nella finestra in cui almeno survival_frac degli episodi e' vivo,
    # calcolati solo sugli episodi di test
    X, y, t, episode = data["X"], data["y"], data["t"], data["episode"]

    ep_len = {}
    for e, ti in zip(episode[is_test], t[is_test]):
        ep_len[e] = max(ep_len.get(e, 0), ti)
    lengths = np.array(list(ep_len.values()))

    cutoff = np.quantile(lengths, 1 - survival_frac)
    n_alive = int((lengths >= cutoff).sum())
    print(f"\nFinestra sicura: step in [0, {cutoff:.0f}]  "
          f"({n_alive}/{len(lengths)} episodi di test ancora vivi, "
          f"{100 * n_alive / len(lengths):.0f}%)")

    mask = is_test & (t <= cutoff)
    X_w, y_w, t_w = X[mask], y[mask], t[mask]

    pred = predict_denorm(model, X_w, mu, sd, device)
    err = np.abs(pred - y_w)

    baseline_mae = np.abs(y[~is_test].mean(0, keepdims=True) - y_w).mean(0)

    edges = np.linspace(0, cutoff, n_buckets + 1)
    print(f"MAE nei primi {cutoff:.0f} step, per massa (solo episodi di test):")
    stats = {
        "cutoff_step": float(cutoff),
        "n_episodes_alive_pct": 100 * n_alive / len(lengths),
        "per_mass": {},
    }
    for mi, name in enumerate(MASS_NAMES):
        print(f"  {name} (baseline naive = {baseline_mae[mi]:.4f}):")
        buckets = []
        for i in range(n_buckets):
            lo, hi = edges[i], edges[i + 1]
            bmask = (t_w >= lo) & (t_w <= hi) if i == n_buckets - 1 else (t_w >= lo) & (t_w < hi)
            if bmask.sum() == 0:
                continue
            mae_val = float(err[bmask, mi].mean())
            impr = 100 * (1 - mae_val / baseline_mae[mi])
            print(f"    step [{lo:3.0f},{hi:3.0f}]: MAE={mae_val:.4f}  "
                  f"({impr:+.1f}% vs baseline, n={bmask.sum()})")
            buckets.append({"lo": float(lo), "hi": float(hi), "mae": mae_val,
                            "improv_pct": float(impr), "n": int(bmask.sum())})

        r_p, p_p = pearsonr(t_w, err[:, mi])
        print(f"    correlazione step/errore nella finestra: r={r_p:.3f}  p={p_p:.4g}")
        stats["per_mass"][name] = {
            "baseline_mae": float(baseline_mae[mi]),
            "buckets": buckets,
            "pearson_r": float(r_p),
            "pearson_p": float(p_p),
        }
    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--hidden_dim", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--early_buckets", type=int, default=5)
    p.add_argument("--survival_frac", type=float, default=0.9)
    p.add_argument("--out", type=str, default="./results/probe_robustness_results.json")
    args = p.parse_args()

    data = np.load(args.dataset)
    device = "cpu"
    print(f"Dataset: {args.dataset}  ({data['X'].shape[0]} righe, "
          f"{len(np.unique(data['episode']))} episodi, feature dim {data['X'].shape[1]})")
    print_metadata(data)

    print(f"\n=== Cross-validation su {args.folds} split degli episodi ===")
    improvs, corrs = cross_validate(data, args.folds, args.epochs, args.hidden_dim,
                                    args.lr, args.batch_size, device, args.seed)

    print("\n=== Finestra iniziale (nessun survivorship bias, solo test set) ===")
    folds = make_episode_folds(data["episode"], args.folds, args.seed)
    is_test = np.isin(data["episode"], folds[0])
    model, mu, sd = train_one_fold(data["X"][~is_test], data["y"][~is_test],
                                   args.hidden_dim, args.epochs, args.lr,
                                   args.batch_size, device)
    early = early_window_analysis(data, model, mu, sd, device, args.early_buckets,
                                  args.survival_frac, is_test)

    out = {
        "dataset": args.dataset,
        "provenance": {k[5:]: str(data[k]) for k in data.files
                       if k.startswith("meta_")},
        "cross_validation": {
            name: {
                "improv_pct_mean": float(improvs[:, i].mean()),
                "improv_pct_std": float(improvs[:, i].std()),
                "pearson_r_mean": float(corrs[:, i].mean()),
                "pearson_r_std": float(corrs[:, i].std()),
            }
            for i, name in enumerate(MASS_NAMES)
        },
        "early_window": early,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=4)
    print(f"\nRisultati salvati in: {args.out}")


if __name__ == "__main__":
    main()
