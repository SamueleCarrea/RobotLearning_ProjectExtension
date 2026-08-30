"""Prints the provenance metadata of every probe dataset (.npz) in a folder.

Why this exists: the trained dataset and the reservoir control dataset must
come from the SAME acting policy and the SAME episode seed, otherwise they are
not collected on the same trajectories and the comparison is not paired.
This script makes that verifiable in one command instead of trusting the
README.

Usage:
    python check_dataset_meta.py
    python check_dataset_meta.py --glob "probe_dataset_*.npz"
"""

import argparse
import glob
import os

import numpy as np

FIELDS = [
    "control_mode",
    "checkpoint",
    "vecnormalize",
    "env_id",
    "seed",
    "encoder_seed",
    "lstm_hidden_size",
    "use_cell_state",
    "deterministic",
    "stride",
]


def read_meta(path):
    d = np.load(path, allow_pickle=True)
    meta = {}
    for f in FIELDS:
        key = "meta_" + f
        meta[f] = str(d[key]) if key in d.files else None
    meta["_rows"] = int(d["X"].shape[0])
    meta["_dim"] = int(d["X"].shape[1])
    meta["_episodes"] = int(len(np.unique(d["episode"])))
    meta["_mean_len"] = float(d["episode_length"].mean())
    return meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--glob", type=str, default="*.npz",
                   help="pattern of the .npz files to inspect")
    args = p.parse_args()

    paths = sorted(glob.glob(args.glob))
    if not paths:
        print(f"No file matches {args.glob}")
        return

    metas = {}
    for path in paths:
        try:
            metas[path] = read_meta(path)
        except Exception as exc:  # noqa: BLE001
            print(f"{path}: cannot read ({exc})")

    for path, m in metas.items():
        print("=" * 72)
        print(os.path.basename(path))
        print(f"  rows={m['_rows']}  feature_dim={m['_dim']}  "
              f"episodes={m['_episodes']}  mean_ep_len={m['_mean_len']:.0f}")
        for f in FIELDS:
            if m[f] is not None:
                print(f"  {f:18s} = {m[f]}")
        if all(m[f] is None for f in FIELDS):
            print("  !! no metadata: this dataset predates the metadata patch,")
            print("     its provenance cannot be verified. Recollect it.")

    # pairing check: every control dataset must match a trained dataset on
    # acting policy and episode seed
    print("\n" + "=" * 72)
    print("PAIRING CHECK (control vs trained)")
    trained = {k: v for k, v in metas.items() if not v["control_mode"]}
    control = {k: v for k, v in metas.items() if v["control_mode"]}
    if not control:
        print("  no control dataset found, nothing to check")
        return
    def norm(path):
        p = (path or "").strip()
        return p[:-4] if p.endswith(".zip") else p

    for cpath, c in control.items():
        # meta_checkpoint of a control is "CONTROL[mode] actor=<path>"
        actor = norm((c["checkpoint"] or "").split("actor=")[-1])
        matches = [os.path.basename(tp) for tp, t in trained.items()
                   if norm(t["checkpoint"]) == actor and t["seed"] == c["seed"]]
        status = "OK  paired with " + ", ".join(matches) if matches else \
                 "!!  NO trained dataset with the same actor AND episode seed"
        print(f"  {os.path.basename(cpath):42s} {status}")
    print("\nIf a control is unpaired, the 'same trajectories' claim does not")
    print("hold for it and it must be recollected with the matching")
    print("--checkpoint and --seed.")


if __name__ == "__main__":
    main()
