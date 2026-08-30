#!/usr/bin/env bash
# Pipeline completa. Un run da 1M timesteps con n_envs=8 sono circa 30-50 min
# su CPU, quindi in totale sono ore: conviene lanciarlo a pezzi.
set -euo pipefail

SEEDS="${SEEDS:-42 123 7}"
TIMESTEPS="${TIMESTEPS:-1000000}"
NENVS="${NENVS:-8}"

for s in $SEEDS; do
  python train.py --env CustomHopper-source-v0 --seed "$s" --n_envs "$NENVS" \
      --timesteps "$TIMESTEPS" --tag "source_s$s"
  python train.py --env CustomHopper-target-v0 --seed "$s" --n_envs "$NENVS" \
      --timesteps "$TIMESTEPS" --tag "target_s$s"
  python train_udr.py --seed "$s" --n_envs "$NENVS" --timesteps "$TIMESTEPS" --tag "s$s"
done

for e in udr5 udr25 udr50; do
  python train_udr.py --udr_env "CustomHopper-source-$e-v0" --seed 42 \
      --n_envs "$NENVS" --timesteps "$TIMESTEPS" --tag "${e}_s42"
done

python train_oracle.py --seed 42 --n_envs "$NENVS" --timesteps "$TIMESTEPS" --tag s42
python train_oracle.py --seed 42 --n_envs "$NENVS" --timesteps "$TIMESTEPS" \
    --oracle_masses all --tag all_s42

python collect_data.py --checkpoint models/udr_s42/udr_RecurrentPPO_final.zip \
    --episodes 450 --use_cell_state --out probe_dataset_450.npz
python train_probe.py --dataset probe_dataset_450.npz --epochs 150
python analyze_probe_robustness.py --dataset probe_dataset_450.npz --folds 5
python analyze_probe_controls.py --dataset probe_dataset_450.npz

python summarize_results.py --markdown results/summary.md
python plot.py
