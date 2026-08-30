"""Allena il ProbeMLP sul dataset di collect_data.py.

Quello che guardiamo non e' il MAE assoluto ma quanto scende rispetto a una
baseline che predice sempre la massa media. Analisi piu' approfondite in
analyze_probe_robustness.py e analyze_probe_controls.py.
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
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


def split_by_episode(episode, test_frac=0.2, seed=0):
    # split per episodio intero, altrimenti step dello stesso episodio
    # finiscono sia in train che in test
    rng = np.random.default_rng(seed)
    eps = np.unique(episode)
    rng.shuffle(eps)
    test_eps = eps[: max(1, int(len(eps) * test_frac))]
    is_test = np.isin(episode, test_eps)
    return ~is_test, is_test


def train_probe(X_tr, y_tr, hidden_dim, epochs, lr, batch_size, device, verbose=True):
    # ritorna anche media e std dei target, servono per de-normalizzare
    mu, sd = y_tr.mean(0), y_tr.std(0) + 1e-8
    model = ProbeMLP(X_tr.shape[1], hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(
            torch.tensor(X_tr, dtype=torch.float32),
            torch.tensor((y_tr - mu) / sd, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
        if verbose and ((epoch + 1) % 25 == 0 or epoch == 0):
            print(f"  epoch {epoch + 1}/{epochs}  loss={total / len(loader.dataset):.4f}")
    return model, mu, sd


def predict_denorm(model, X, mu, sd, device):
    model.eval()
    with torch.no_grad():
        pred_n = model(torch.tensor(X, dtype=torch.float32, device=device)).cpu().numpy()
    return pred_n * sd + mu


def report_buckets(err, values, edges, label, baseline_mae, fmt="{:.0f}"):
    # MAE per bucket, riportato come miglioramento % sulla baseline
    print(f"\nMAE per {label}, per massa:")
    out = {}
    for mi, name in enumerate(MASS_NAMES):
        print(f"  {name}:")
        rows = []
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            last = i == len(edges) - 2
            m = (values >= lo) & (values <= hi) if last else (values >= lo) & (values < hi)
            if m.sum() == 0:
                continue
            mae = float(err[m, mi].mean())
            impr = 100 * (1 - mae / baseline_mae[mi])
            print(f"    [{fmt.format(lo)}, {fmt.format(hi)}]: MAE={mae:.4f} "
                  f"({impr:+.1f}% vs baseline, n={m.sum()})")
            rows.append({"lo": float(lo), "hi": float(hi), "mae": mae,
                         "improv_pct": float(impr), "n": int(m.sum())})
        out[name] = rows
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="probe_dataset_450.npz")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden_dim", type=int, default=32,
                   help="tenerlo piccolo, il probe deve restare debole")
    p.add_argument("--test_frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_buckets", type=int, default=4)
    p.add_argument("--out_model", type=str, default="probe_mlp.pt")
    p.add_argument("--out_json", type=str, default="./results/probe_results.json")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = np.load(args.dataset)
    print(f"Dataset: {args.dataset}  ({data['X'].shape[0]} righe, "
          f"{len(np.unique(data['episode']))} episodi, feature dim {data['X'].shape[1]})")
    print_metadata(data)

    is_tr, is_te = split_by_episode(data["episode"], args.test_frac, args.seed)
    X_tr, y_tr = data["X"][is_tr], data["y"][is_tr]
    X_te, y_te, t_te, ep_te = (data["X"][is_te], data["y"][is_te],
                               data["t"][is_te], data["episode"][is_te])

    print(f"\nTrain: {X_tr.shape[0]} righe  |  Test: {X_te.shape[0]} righe  |  device={device}")
    model, mu, sd = train_probe(X_tr, y_tr, args.hidden_dim, args.epochs,
                                args.lr, args.batch_size, device)

    pred = predict_denorm(model, X_te, mu, sd, device)
    err = np.abs(pred - y_te)
    probe_mae = err.mean(0)
    baseline_mae = np.abs(y_tr.mean(0, keepdims=True) - y_te).mean(0)

    print("\nRisultato principale (test set, valori in kg):")
    results = {"dataset": args.dataset, "per_mass": {}}
    for i, name in enumerate(MASS_NAMES):
        impr = 100 * (1 - probe_mae[i] / baseline_mae[i])
        print(f"  {name}: baseline={baseline_mae[i]:.4f}  probe={probe_mae[i]:.4f}  "
              f"miglioramento={impr:+.1f}%")
        results["per_mass"][name] = {
            "baseline_mae": float(baseline_mae[i]),
            "probe_mae": float(probe_mae[i]),
            "improv_pct": float(impr),
        }

    print("\nCorrelazione timestep / errore assoluto:")
    for i, name in enumerate(MASS_NAMES):
        r_p, p_p = pearsonr(t_te, err[:, i])
        r_s, p_s = spearmanr(t_te, err[:, i])
        print(f"  {name}: Pearson r={r_p:+.3f} p={p_p:.3g}  |  Spearman r={r_s:+.3f} p={p_s:.3g}")
        results["per_mass"][name].update({"pearson_r": float(r_p), "pearson_p": float(p_p),
                                          "spearman_r": float(r_s), "spearman_p": float(p_s)})

    edges = np.quantile(t_te, np.linspace(0, 1, args.n_buckets + 1))
    results["by_timestep"] = report_buckets(err, t_te, edges, "fase assoluta (step)",
                                            baseline_mae)

    # fase relativa, corregge per la diversa lunghezza degli episodi
    if "episode_length" in data.files:
        ep_len = data["episode_length"][is_te].astype(float)
    else:
        lut = {}
        for e, ti in zip(ep_te, t_te):
            lut[e] = max(lut.get(e, 0), ti)
        ep_len = np.array([max(lut[e], 1) for e in ep_te], dtype=float)
    rel_t = t_te / np.maximum(ep_len, 1)
    edges_r = np.quantile(rel_t, np.linspace(0, 1, args.n_buckets + 1))
    results["by_relative_phase"] = report_buckets(err, rel_t, edges_r,
                                                  "fase relativa dell'episodio",
                                                  baseline_mae, fmt="{:.2f}")

    torch.save({"state_dict": model.state_dict(), "y_mean": mu, "y_std": sd,
                "input_dim": X_tr.shape[1], "hidden_dim": args.hidden_dim},
               args.out_model)
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nModello salvato in {args.out_model}  |  metriche in {args.out_json}")


if __name__ == "__main__":
    main()
