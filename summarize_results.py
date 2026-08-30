"""Aggrega i JSON in results/ e stampa media e std calcolate tra i seed.

La std dentro un singolo run e' quella tra episodi di valutazione ed e' molto
piccola; per confrontare due metodi serve la variabilita' tra run diversi.
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np


def load_all(results_dir):
    runs = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*_results.json"))):
        name = os.path.basename(path)
        if name.startswith("probe_"):
            continue  # i risultati del probe hanno una struttura diversa
        with open(path) as f:
            d = json.load(f)
        if "results" not in d or "source_to_target" not in d.get("results", {}):
            continue
        runs.append((name, d))
    return runs


def group_key(d):
    cfg = d.get("config")
    method = d.get("training_method", d.get("training_env", "?"))
    env = d.get("training_env", "?")
    ts = d.get("timesteps", "?")
    if cfg is None:
        return (method, env, ts, "legacy (no config, no VecNormalize)")
    desc = (f"n_envs={cfg.get('n_envs')}, normalize={cfg.get('normalize')}, "
            f"lstm={cfg.get('lstm_hidden_size') or '-'}")
    if "oracle_masses" in cfg:
        desc += f", masses={cfg['oracle_masses']}"
    return (method, env, ts, desc)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", type=str, default="./results")
    p.add_argument("--markdown", type=str, default=None,
                   help="Se indicato, scrive anche una tabella markdown pronta per il report.")
    args = p.parse_args()

    runs = load_all(args.results_dir)
    if not runs:
        print(f"Nessun risultato trovato in {args.results_dir}")
        return

    groups = defaultdict(list)
    for name, d in runs:
        groups[group_key(d)].append((name, d))

    rows = []
    for key, items in sorted(groups.items()):
        method, env, ts, desc = key
        src = np.array([d["results"]["source_to_source"]["mean"] for _, d in items])
        tgt = np.array([d["results"]["source_to_target"]["mean"] for _, d in items])
        seeds = [d.get("config", {}).get("seed") for _, d in items]
        rows.append({
            "method": method, "env": env, "timesteps": ts, "config": desc,
            "n_runs": len(items),
            "seeds": seeds,
            "source_mean": float(src.mean()), "source_std": float(src.std(ddof=0)),
            "target_mean": float(tgt.mean()), "target_std": float(tgt.std(ddof=0)),
            "files": [n for n, _ in items],
        })

    rows.sort(key=lambda r: -r["target_mean"])

    print("=" * 100)
    print(f"{'metodo / config':<52}{'n':>3}  {'-> source':>16}  {'-> target':>16}")
    print("=" * 100)
    for r in rows:
        label = f"{r['method']} [{r['config']}] {r['timesteps']}"
        print(f"{label[:52]:<52}{r['n_runs']:>3}  "
              f"{r['source_mean']:>8.1f} ±{r['source_std']:>6.1f}  "
              f"{r['target_mean']:>8.1f} ±{r['target_std']:>6.1f}")
        if r["n_runs"] < 3:
            print(f"{'':<52}     ATTENZIONE: solo {r['n_runs']} run, "
                  f"la std tra seed non e' affidabile")
    print("=" * 100)
    print("Le deviazioni standard sono calcolate TRA i run (seed diversi), "
          "non tra episodi di valutazione.")

    out_path = os.path.join(args.results_dir, "summary.json")
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=4)
    print(f"\nRiepilogo salvato in: {out_path}")

    if args.markdown:
        lines = ["| Metodo | Env | Config | Run | → source | → target |",
                 "|---|---|---|---:|---:|---:|"]
        for r in rows:
            n_flag = f"{r['n_runs']}" if r["n_runs"] >= 3 else f"{r['n_runs']} (*)"
            lines.append(
                f"| {r['method']} | {r['env']} | {r['config']} | {n_flag} | "
                f"{r['source_mean']:.1f} ± {r['source_std']:.1f} | "
                f"{r['target_mean']:.1f} ± {r['target_std']:.1f} |"
            )
        lines.append("")
        lines.append("(*) un solo seed: la std non e' affidabile, e' solo quella tra episodi di eval")
        with open(args.markdown, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"Tabella markdown salvata in: {args.markdown}")


if __name__ == "__main__":
    main()