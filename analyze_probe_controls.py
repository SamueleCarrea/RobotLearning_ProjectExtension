"""Controlli per verificare che il probe non stia barando.

1. Survivorship: episodi con certe masse muoiono prima, quindi le masse
   correlano col timestep. Controllo: probe con solo t come input.
2. Leakage: controllo con le masse permutate tra episodi.

In piu' il miglioramento per finestra temporale. A t=0 deve essere ~0, la
fisica non ha ancora prodotto informazione sulle masse.
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from torch.utils.data import DataLoader, TensorDataset

from probe_model import ProbeMLP

MASS_NAMES = ["thigh", "leg", "foot"]
WINDOWS = [(0, 0), (1, 20), (20, 50), (50, 100), (100, 200), (200, 500)]


def fit_probe(X_tr, y_tr, X_te, hidden_dim, epochs, lr, batch_size, device):
    # allena il probe e restituisce le predizioni in kg
    mu, sd = y_tr.mean(0), y_tr.std(0) + 1e-8
    net = ProbeMLP(X_tr.shape[1], hidden_dim).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(
            torch.tensor(X_tr, dtype=torch.float32),
            torch.tensor((y_tr - mu) / sd, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    for _ in range(epochs):
        net.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(net(xb), yb).backward()
            opt.step()

    net.eval()
    with torch.no_grad():
        pred_n = net(torch.tensor(X_te, dtype=torch.float32, device=device)).cpu().numpy()
    return pred_n * sd + mu


def fit_linear_probe(X_tr, y_tr, X_te, alpha=10.0):
    # se funziona anche il lineare vuol dire che l'informazione e'
    # direttamente decodificabile, non solo presente
    mu, sd = y_tr.mean(0), y_tr.std(0) + 1e-8
    # standardizzo le feature col train, altrimenti la ridge e' instabile
    xm, xs = X_tr.mean(0), X_tr.std(0) + 1e-8
    reg = Ridge(alpha=alpha).fit((X_tr - xm) / xs, (y_tr - mu) / sd)
    return reg.predict((X_te - xm) / xs) * sd + mu


def improvement_pct(pred, y_te, y_tr):
    # miglioramento % del MAE sulla baseline (media del train)
    base = np.abs(y_tr.mean(0, keepdims=True) - y_te).mean(0)
    probe = np.abs(pred - y_te).mean(0)
    return 100.0 * (1.0 - probe / base), base


def split_by_episode(episode, test_frac, seed):
    rng = np.random.default_rng(seed)
    eps = np.unique(episode)
    rng.shuffle(eps)
    test_eps = set(eps[: max(1, int(len(eps) * test_frac))].tolist())
    is_test = np.array([e in test_eps for e in episode])
    return ~is_test, is_test


def shuffle_labels_across_episodes(y, episode, seed):
    # permuta le masse tra episodi, coerenti dentro l'episodio
    rng = np.random.default_rng(seed + 1)
    per_ep = {e: y[episode == e][0] for e in np.unique(episode)}
    keys = list(per_ep.keys())
    vals = [per_ep[k] for k in keys]
    rng.shuffle(vals)
    remap = dict(zip(keys, vals))
    return np.stack([remap[e] for e in episode])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="probe_dataset_450.npz")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--hidden_dim", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="./results/probe_controls_results.json")
    args = ap.parse_args()

    device = "cpu"
    d = np.load(args.dataset)
    X, y, t, episode = d["X"], d["y"], d["t"], d["episode"]
    print(f"Dataset: {args.dataset}  ({X.shape[0]} righe, {len(np.unique(episode))} episodi, "
          f"feature dim = {X.shape[1]})")
    for k in sorted(k for k in d.files if k.startswith("meta_")):
        print(f"  {k[5:]:18s} = {d[k]}")

    is_tr, is_te = split_by_episode(episode, args.test_frac, args.seed)
    y_tr, y_te, t_te = y[is_tr], y[is_te], t[is_te]

    kw = dict(hidden_dim=args.hidden_dim, epochs=args.epochs, lr=args.lr,
              batch_size=args.batch_size, device=device)
    # provenance: without this the output JSON cannot be traced back to the
    # policy it came from, which is exactly the ambiguity we hit once already
    out = {
        "dataset": args.dataset,
        "provenance": {k[5:]: str(d[k]) for k in d.files if k.startswith("meta_")},
    }

    # probe vero
    pred = fit_probe(X[is_tr], y_tr, X[is_te], **kw)
    improv, base = improvement_pct(pred, y_te, y_tr)
    out["hidden_state"] = dict(zip(MASS_NAMES, improv.tolist()))

    # probe lineare
    pred_lin = fit_linear_probe(X[is_tr], y_tr, X[is_te])
    improv_lin, _ = improvement_pct(pred_lin, y_te, y_tr)
    out["linear_ridge_probe"] = dict(zip(MASS_NAMES, improv_lin.tolist()))

    # controllo 1: solo il timestep
    T = np.stack([t / 500.0, (t / 500.0) ** 2, np.ones_like(t)], 1).astype(np.float32)
    pred_t = fit_probe(T[is_tr], y_tr, T[is_te], **kw)
    improv_t, _ = improvement_pct(pred_t, y_te, y_tr)
    out["control_timestep_only"] = dict(zip(MASS_NAMES, improv_t.tolist()))

    # controllo 2: etichette permutate
    y_sh = shuffle_labels_across_episodes(y, episode, args.seed)
    pred_sh = fit_probe(X[is_tr], y_sh[is_tr], X[is_te], **kw)
    improv_sh, _ = improvement_pct(pred_sh, y_sh[is_te], y_sh[is_tr])
    out["control_shuffled_labels"] = dict(zip(MASS_NAMES, improv_sh.tolist()))

    print("\nMiglioramento % del MAE rispetto alla baseline naive:")
    for label, vals in (("MLP su hidden state", improv),
                        ("Ridge lineare      ", improv_lin),
                        ("CONTROLLO solo t   ", improv_t),
                        ("CONTROLLO shuffled ", improv_sh)):
        print(f"  {label}: " + "   ".join(f"{n}={v:+.1f}%" for n, v in zip(MASS_NAMES, vals)))

    # andamento temporale
    err = np.abs(pred - y_te)
    print("\nMiglioramento % per finestra temporale (t=0 deve essere ~0%):")
    out["by_window"] = {}
    for lo, hi in WINDOWS:
        m = (t_te >= lo) & (t_te <= hi)
        if m.sum() < 20:
            continue
        w = 100.0 * (1.0 - err[m].mean(0) / base)
        out["by_window"][f"{lo}-{hi}"] = {"n": int(m.sum()), **dict(zip(MASS_NAMES, w.tolist()))}
        print(f"  t in [{lo:3d},{hi:3d}]  n={m.sum():5d}: "
              + "   ".join(f"{n}={v:+.1f}%" for n, v in zip(MASS_NAMES, w)))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=4)
    print(f"\nRisultati salvati in: {args.out}")


if __name__ == "__main__":
    main()
